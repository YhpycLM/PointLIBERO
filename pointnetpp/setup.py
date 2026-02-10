import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["torch>=1.4"]

# ==================== 【核心修改】: 更新支持的GPU架构列表 ====================
# 我们保留了旧的架构以实现向后兼容，并添加了对新架构的支持。
#
#   - 7.0, 7.5: Volta (V100), Turing (RTX 20xx)
#   - 8.0, 8.6, 8.7: Ampere (A100, RTX 30xx)
#   - 8.9: Ada Lovelace (RTX 40xx)
#   - 9.0: Hopper (H100)
#
# "+PTX" 表示同时编译一个向前兼容的中间版本。
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
# ===========================================================================

# 尝试打开 _version.py，如果不存在则设置一个默认版本
try:
    exec(open("_version.py").read())
except FileNotFoundError:
    __version__ = "1.0.0"  # 设置一个默认值

setup(
    name='pointnet2',
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(_this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)



