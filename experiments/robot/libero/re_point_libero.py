"""
通过在环境中重放演示来重新生成LIBERO数据集（HDF5文件），并集成了点云生成功能。

注意：
    - 图像观测值以 256x256 分辨率保存。
    - 过滤掉不会改变机器人状态的“无操作”（零）动作的转换。
    - 过滤掉不成功的演示。
    - 在每个成功的时间步，从'agentview'相机生成裁剪后的点云，并将其存储在新的HDF5文件中。
    - 所有点云都被采样或填充到固定的点数（N_POINTS），以便于批处理。

用法：
    python regenerate_libero_dataset_with_pointcloud.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

示例 (LIBERO-Spatial):
    python regenerate_libero_dataset_with_pointcloud.py \
        --libero_task_suite libero_spatial \
        --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
        --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_pcd
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
import open3d as o3d
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix
# ================= 新增导入 =================
from scipy.spatial import cKDTree
# ========================================

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from typing import List


def process_and_save_point_cloud(
        main_cloud_numpy: np.ndarray,
        gripper_pose: np.ndarray,
        output_path: str,
        radius: float = 0.02,
        color: list = [1.0, 0.0, 0.0]
) -> None:
    """
    接收一个NumPy数组形式的点云，将夹爪位置可视化为小球，并将结果保存到本地.ply文件。

    Args:
        main_cloud_numpy (np.ndarray): 背景点云，一个形状为 (N, 3) 或 (N, 6) 的 NumPy 数组。
        gripper_pose (np.ndarray): 夹爪的6D位姿 (x, y, z, ax, ay, az)。
        output_path (str): 保存 .ply 文件的完整路径。
        radius (float, optional): 可视化小球的半径。默认为 0.02。
        color (list, optional): 小球的 RGB 颜色。默认为红色 [1.0, 0.0, 0.0]。
    """
    # ==================== 内部转换步骤 ====================
    # 1. 将输入的 NumPy 数组转换为 Open3D PointCloud 对象
    main_pcd = o3d.geometry.PointCloud()

    if main_cloud_numpy.shape[1] == 6:
        # 如果是 6 维 (XYZ + RGB)
        main_pcd.points = o3d.utility.Vector3dVector(main_cloud_numpy[:, :3])
        main_pcd.colors = o3d.utility.Vector3dVector(main_cloud_numpy[:, 3:])
    elif main_cloud_numpy.shape[1] == 3:
        # 如果只有 3 维 (XYZ)
        main_pcd.points = o3d.utility.Vector3dVector(main_cloud_numpy)
        # 给一个默认的灰色背景以便观察
        main_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    else:
        raise ValueError("输入的 NumPy 数组维度不正确，应为 (N, 3) 或 (N, 6)")
    # =======================================================

    # 2. 从6D位姿中提取XYZ位置
    gripper_position = gripper_pose[:3]
    print(f"夹爪位置 (x, y, z): {gripper_position}")

    # 3. 创建一个代表夹爪位置的小球
    gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)
    gripper_sphere.translate(gripper_position)
    gripper_sphere.paint_uniform_color(color)

    # 4. 将小球的网格顶点转换为点云格式，以便合并
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = gripper_sphere.vertices
    sphere_pcd.colors = gripper_sphere.vertex_colors

    # 5. 合并背景点云和夹爪小球的点云
    combined_pcd = main_pcd + sphere_pcd

    # 6. 将合并后的点云保存到本地文件
    print(f"正在将合并后的点云保存到: '{output_path}'...")
    success = o3d.io.write_point_cloud(output_path, combined_pcd)

    if success:
        print(f"✅ 点云已成功保存！")
        # (可选) 同时进行可视化预览
        print("正在打开可视化窗口进行预览... 按 'q' 关闭。")
        o3d.visualization.draw_geometries([combined_pcd], window_name="点云与夹爪位置")
    else:
        print(f"❌ 错误：保存点云失败。")

def save_ply(points: np.ndarray, path: str):
    assert isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 6
    xyz = points[:, :3].astype(np.float32)
    rgb = points[:, 3:6].astype(np.float32)

    # 颜色兼容：支持 0–255 或 0–1
    if np.nanmax(rgb) > 1.0:
        rgb = np.clip(rgb, 0, 255) / 255.0
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    # 去掉 NaN/Inf
    mask = np.isfinite(xyz).all(1) & np.isfinite(rgb).all(1)
    xyz, rgb = xyz[mask], rgb[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    ok = o3d.io.write_point_cloud(path, pcd)  # 后缀 .ply
    if not ok:
        raise RuntimeError(f"写入失败：{path}")

# --- 新增的常量 ---
IMAGE_RESOLUTION = 256
# 为每个点云设置一个固定的点数，便于深度学习模型处理
N_POINTS = 4096

# ===============================================================================
# 函数集成自 代码A: 用于点云生成和处理
# ===============================================================================

def depth_to_pointcloud(depth, rgb, intrinsics_inv, extrinsics_inv):
    """
    将深度图和RGB图像转换为点云。
    """
    h, w = depth.shape
    i, j = np.mgrid[0:h, 0:w]
    pixels = np.stack([j, i, np.ones_like(depth)], axis=-1).reshape(-1, 3)
    depth_flat = depth.flatten()

    points_cam = (pixels * depth_flat[:, np.newaxis]) @ intrinsics_inv.T
    points_cam_homo = np.hstack([points_cam, np.ones((h * w, 1))])

    points_world_homo = points_cam_homo @ extrinsics_inv.T
    points = points_world_homo[:, :3]

    colors = rgb.reshape(-1, 3) / 255.0
    valid_indices = (depth_flat > 0.1) & (depth_flat < 2.0)
    return points[valid_indices], colors[valid_indices]


# ================= 新增的重采样函数 =================
def resample_pointcloud(pcd_data: np.ndarray, n_points: int) -> np.ndarray:
    """
    将点云重采样到固定数量的点。
    - 如果点数过多，则使用最远点采样 (FPS) 进行下采样。
    - 如果点数过少，则通过随机重复进行上采样（填充）。

    Args:
        pcd_data (np.ndarray): 输入点云，形状为 (M, 6)。
        n_points (int): 目标点数。

    Returns:
        np.ndarray: 重采样后的点云，形状为 (n_points, 6)。
    """
    num_current_points = pcd_data.shape[0]

    if num_current_points == n_points:
        return pcd_data

    elif num_current_points > n_points:
        # --- 最远点采样 (FPS) ---
        # 1. 分离XYZ坐标用于几何采样
        xyz = pcd_data[:, :3]

        # 2. 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # 3. 执行最远点采样
        downsampled_pcd = pcd.farthest_point_down_sample(n_points)
        downsampled_xyz = np.asarray(downsampled_pcd.points)

        # 4. 使用KDTree高效恢复完整的6维特征
        kdtree = cKDTree(xyz)
        _, original_indices = kdtree.query(downsampled_xyz, k=1)

        final_pcd = pcd_data[original_indices]
        return final_pcd

    else:  # num_current_points < n_points
        # --- 上采样（填充）---
        # 随机选择一些点进行复制，以达到目标数量
        indices_to_add = np.random.choice(num_current_points, n_points - num_current_points, replace=True)
        padding_data = pcd_data[indices_to_add, :]
        final_pcd = np.vstack((pcd_data, padding_data))
        return final_pcd
# ===================================================


def generate_and_process_pcd(env, obs, base_pos, n_points):
    """
    一个辅助函数，用于处理观测数据，生成点云，根据基座位置固定裁剪，
    并采样/填充到固定大小，最终返回一个NumPy数组。

    Args:
        env: 仿真环境实例。
        obs (dict): 当前时间步的观测字典。
        base_pos (np.ndarray): 机器人基座的世界坐标 [x, y, z]。
        n_points (int): 输出点云应包含的点数。

    Returns:
        np.ndarray: 形状为 (n_points, 6) 的数组，每一行是 [x, y, z, r, g, b]。
    """
    sim = env.sim
    # 假设我们总是使用 'agentview' 相机
    camera_name = "agentview"

    # 从观测数据中获取图像
    rgb_img = obs[f"{camera_name}_image"][::-1, :, :]
    norm_depth = obs[f"{camera_name}_depth"][::-1]
    real_depth = get_real_depth_map(sim, norm_depth)
    if real_depth.ndim > 2:
        real_depth = np.squeeze(real_depth, axis=-1)

    # 获取相机内外参
    h, w = real_depth.shape
    intrinsics = get_camera_intrinsic_matrix(sim, camera_name, h, w)
    intrinsics_inv = np.linalg.inv(intrinsics)

    # 计算相机外参逆矩阵
    cam_pos_in_world = sim.data.get_camera_xpos(camera_name)
    cam_rot_in_world = sim.data.get_camera_xmat(camera_name)
    cv_to_mujoco_cam_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    cv_cam_rot_in_world = cam_rot_in_world @ cv_to_mujoco_cam_rotation
    extrinsics_inv_manual = np.eye(4)
    extrinsics_inv_manual[:3, :3] = cv_cam_rot_in_world
    extrinsics_inv_manual[:3, 3] = cam_pos_in_world

    # 生成原始点云
    points, colors = depth_to_pointcloud(real_depth, rgb_img, intrinsics_inv, extrinsics_inv_manual)


    if points.size == 0:
        # 如果没有有效的点，返回一个全零数组
        return np.zeros((n_points, 6), dtype=np.float32)

    # --- 固定的裁剪逻辑 ---
    if base_pos is not None:
        # 裁剪掉机器人基座后方的点
        cropping_plane_x = base_pos[0]
        cropping_mask = points[:, 0] > cropping_plane_x
        points_cropped = points[cropping_mask]
        colors_cropped = colors[cropping_mask]
    else:
        points_cropped = points
        colors_cropped = colors

    if points_cropped.shape[0] == 0:
        return np.zeros((n_points, 6), dtype=np.float32)

    # 合并点和颜色
    pcd_data = np.hstack((points_cropped, colors_cropped))
    # save_ply(pcd_data, '/home/lm/Desktop/openvla-main/assets/shuchu_ceshi/total.ply')
    # ================= 替换原有逻辑 =================
    # --- 使用新的重采样函数，将点云采样或填充到 n_points ---
    final_pcd = resample_pointcloud(pcd_data, n_points)
    # save_ply(final_pcd, '/home/lm/Desktop/openvla-main/assets/shuchu_ceshi/fps.ply')
    a = 1
    # ===============================================

    return final_pcd.astype(np.float32)


# ===============================================================================
# 函数来自 代码B: 用于数据集处理 (此部分无需改动)
# ===============================================================================

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    判断一个动作是否为“无操作”。
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset with point clouds!")

    os.makedirs(args.libero_target_dir, exist_ok=True)

    metainfo_json_dict = {}
    metainfo_json_out_path = f"/home/lm/Desktop/openvla-main/experiments/robot/libero/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        json.dump(metainfo_json_dict, f)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            env.reset()
            env.set_init_state(orig_states[0])
            obs = None  # 确保obs被定义
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            sim = env.sim
            base_body_id = sim.model.body_name2id('robot0_base')
            base_pos = sim.data.body_xpos[base_body_id]
            print(f"[信息] 任务 '{task.name}', Demo {i}: 机械臂基座世界坐标: {np.round(base_pos, 3)}")

            states, actions, ee_states, gripper_states, joint_states = [], [], [], [], []
            robot_states, agentview_images, eye_in_hand_images = [], [], []
            pointclouds = []

            for t, action in enumerate(orig_actions):
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    num_noops += 1
                    continue

                if not states:
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                actions.append(action)

                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
                visual = np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                pcd_data = generate_and_process_pcd(env, obs, base_pos, n_points=N_POINTS)
                process_and_save_point_cloud(main_cloud_numpy=pcd_data, gripper_pose=visual, output_path='/media/lm/lmssd1/robot_dataset/libero/point_libero_spatial_rlds/test/1.ply')
                a = 1
                pointclouds.append(pcd_data)

                obs, reward, done, info = env.step(action.tolist())

            if done:
                assert len(actions) == len(agentview_images) == len(pointclouds)

                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))

                obs_grp.create_dataset("point", data=np.stack(pointclouds, axis=0))

                ep_data_grp.create_dataset("actions", data=np.stack(actions))
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )
            print(f"  Total # no-op actions filtered out: {num_noops}")

        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial",
                        required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_pcd",
                        required=True)
    args = parser.parse_args()

    main(args)