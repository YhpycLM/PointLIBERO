"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from pointnetpp.pointnetpp import PointNetPP
from experiments.robot.openvla_utils import get_vla
import numpy as np
import torch
from transformers import AutoModelForVision2Seq
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from pointnetpp.pointnetpp import PointNetPP
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

class OpenVLAForActionPredictionWithPointNet(OpenVLAForActionPrediction):
    """
    一个自定义的模型类，它继承自 OpenVLAForActionPrediction，
    并在初始化时就包含了 PointNet++ 编码器。
    """
    def __init__(self, config: OpenVLAConfig):
        # 首先，调用父类的初始化方法，构建基础的VLA模型
        super().__init__(config)

        # 然后，基于config中的信息，构建并附加pointnet_encoder
        vla_hidden_size = self.config.text_config.hidden_size
        self.pointnet_encoder = PointNetPP(
            sa_n_points=[1024, 256, 64, 16],
            sa_n_samples=[32, 32, 32, 32],
            sa_radii=[0.1, 0.2, 0.4, 0.8],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 512]],
            target_dim=vla_hidden_size,
            return_feature_sequence=True
        )

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# robot_utils.py

# ... (确保顶部的导入是完整的)
import os  # 需要导入 os 模块
import torch
from transformers import AutoModelForVision2Seq
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from pointnetpp.pointnetpp import PointNetPP
from experiments.robot.openvla_utils import get_vla  # 确保 get_vla 被导入


def get_model(cfg, wrap_diffusion_policy_for_droid=False):
    """
    加载模型进行评估。
    此最终版本使用自定义模型类来正确加载包含点云编码器的、分片的模型权重。
    """
    if cfg.model_family == "openvla":
        # 检查是否需要加载带有点云编码器的模型
        if cfg.use_pointcloud:
            print("=" * 80)
            print(">>>>> 正在加载带有 PointNet++ 编码器的 OpenVLA (分片)模型 <<<<<")
            print("=" * 80)

            # 直接使用我们自定义的、架构正确的类来调用 from_pretrained。
            # Hugging Face 会自动处理所有的分片文件 (.safetensors)。
            print(f"从最终 checkpoint '{cfg.pretrained_checkpoint}' 加载模型...")
            model = OpenVLAForActionPredictionWithPointNet.from_pretrained(
                cfg.pretrained_checkpoint,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                # trust_remote_code=True # 如果自定义类和模型代码在同一项目中，通常是安全的
            )
            print("模型权重从所有分片中加载成功。")

            # 将模型移动到评估设备
            model = model.to(DEVICE)

        else:
            # 如果不使用点云，则沿用原始的加载方式
            model = get_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")

    model.eval()  # 设置为评估模式
    print(f"模型加载成功: {type(model)}")

    # 验证 pointnet_encoder 是否存在
    if cfg.use_pointcloud:
        assert hasattr(model, 'pointnet_encoder'), "错误：模型加载后 'pointnet_encoder' 模块丢失！"
        print("'pointnet_encoder' 模块已成功加载并验证。")
        params = list(model.pointnet_encoder.parameters())
        if len(params) > 0:
            # 计算所有参数的绝对值总和
            total_sum = sum(p.abs().sum().item() for p in params)
            print("-" * 50)
            print(f"[DEBUG in get_model] Checking pointnet_encoder weights...")
            print(f"  - Sum of all absolute weights: {total_sum}")
            if total_sum == 0:
                print("  - !!! WARNING: PointNet Encoder weights are all ZERO. !!!")
            else:
                print("  - Weights seem to be loaded correctly.")
            print("-" * 50)

    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
