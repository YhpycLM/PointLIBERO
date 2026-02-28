from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from LIBERO_Spatial.conversion_utils import MultiThreadedDatasetBuilder


# ==================== 简化后的点云预处理函数 ====================
def normalize_point_cloud_for_robotics(
        point_cloud_raw: np.ndarray,
        ee_state_raw: np.ndarray
) -> np.ndarray:
    """
    根据机器人末端执行器状态对点云进行预处理。
    1. 将点云的XYZ坐标转换为相对于夹爪的局部坐标。
    2. （已简化）直接使用已归一化的RGB颜色。

    Args:
        point_cloud_raw (np.ndarray): 原始点云，形状为 (N, 6)，前3列为XYZ，后3列为已归一化的RGB。
        ee_state_raw (np.ndarray): 原始末端执行器状态，形状为 (>=3,)，前3个值为夹爪的XYZ位置。

    Returns:
        np.ndarray: 处理后的点云，形状为 (N, 6)。
    """
    # 1. 提取夹爪的XYZ位置
    gripper_pos = ee_state_raw[:3]

    # 2. 分离点云的XYZ和RGB
    xyz = point_cloud_raw[:, :3]
    rgb = point_cloud_raw[:, 3:]

    # 3. 计算相对于夹爪的局部XYZ坐标
    relative_xyz = xyz - gripper_pos

    # 4. (已简化) 直接使用已归一化的RGB值，仅做安全裁剪
    normalized_rgb = np.clip(rgb, 0.0, 1.0)

    # 5. 重新拼接处理后的特征
    processed_pointcloud = np.hstack([relative_xyz, normalized_rgb])

    return processed_pointcloud


# ===============================================================


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path, demo_id):
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None

            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            ee_states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]
            points_raw = F['data'][f"demo_{demo_id}"]["obs"]["point"][()]

        # compute language instruction
        raw_file_string = os.path.basename(episode_path).split('/')[-1]
        words = raw_file_string[:-10].split("_")
        command = ''
        for w in words:
            if "SCENE" in w:
                command = ''
                continue
            command = command + w + ' '
        command = command[:-1]

        episode = []
        for i in range(actions.shape[0]):
            current_point_cloud_raw = points_raw[i]
            current_ee_state = ee_states[i]

            processed_point_cloud = normalize_point_cloud_for_robotics(
                current_point_cloud_raw,
                current_ee_state
            )

            episode.append({
                'observation': {
                    'image': images[i][::-1, ::-1],
                    'wrist_image': wrist_images[i][::-1, ::-1],
                    'state': np.asarray(np.concatenate((ee_states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                    'point_cloud': np.asarray(processed_point_cloud, dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        return episode_path + f"_{demo_id}", sample

    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F['data'])
        idx = 0
        cnt = 0
        while cnt < n_demos:
            ret = _parse_example(sample, idx)
            if ret is not None:
                cnt += 1
                yield ret
            idx += 1


class LIBEROSpatial(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40
    MAX_PATHS_IN_MEMORY = 80
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        ),
                        'point_cloud': tfds.features.Tensor(
                            shape=(4096, 6),
                            dtype=np.float32,
                            doc='Point cloud from agent view, processed to be relative to the gripper and normalized.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/media/lm/lmssd1/robot_dataset/libero/traget/*.hdf5"),
        }