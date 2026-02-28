<div align="center">

# PointLIBERO: Unlocking Spatial Awareness in VLAs with a Novel 3D Dataset and a Lightweight Framework

**Meng Li** · **Qi Zhao** · [**Shuchang Lyu**](https://cv-shuchanglyu.github.io/EnHome.html) · **Jun Jiang** · **Longhao Zou** · [**Guangliang Cheng**](https://sites.google.com/view/guangliangcheng)

</div>

---

## 📖 Introduction
This repository contains the implementation of **"PointLIBERO: Unlocking Spatial Awareness in VLAs with a Novel 3D Dataset and a Lightweight Framework"**. PointLIBERO aims to enhance Vision-Language-Action models by incorporating 3D spatial awareness.

## 📝 TODO / Roadmap
- [x] Release PointLIBERO Dataset Generation Code
- [ ] Release Checkpoints and Test Code (Expected: 2-3 months)
- [ ] Release Training Code (Expected: 3-4 months)

## 🛠️ Dataset Generation

To generate the PointLIBERO dataset, follow the steps below.

### 1. Environment Setup and Data Preparation
Clone the official LIBERO repository and set up the environment.

```
# Create and activate the conda environment
conda create -n pointlibero python=3.10 -y
conda activate pointlibero

# Clone the original LIBERO repository
cd PointLibero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Install LIBERO dependencies
cd LIBERO
pip install -e .
```
Download raw data from [LIBERO](https://libero-project.github.io/main.html)

### 2. Generate PointLIBERO Data (HDF5)
Select the LIBERO suite you wish to process (e.g., libero_spatial) and run the generation script.
```
# Navigate to the directory
cd pointlibero/experiments/robot/libero

# Run the generation script
# Replace arguments with your actual paths
python re_point_libero.py \
  --libero_task_suite libero_spatial \
  --libero_raw_data_dir /path/to/original/libero/data \
  --libero_target_dir /path/to/save/pointlibero/data
```
### 3. Convert to RLDS Format
The output from the previous step is a standard HDF5 file. To convert this into the universal RLDS format, we recommend using a separate environment to avoid dependency conflicts.
```
# Clone the RLDS dataset builder
cd rlds_trans
git clone https://github.com/kpertsch/rlds_dataset_builder.git

#  Create a separate environment for RLDS
conda env create -f environment_ubuntu.yml
conda activate rlds_env

#  Copy the PointLIBERO builder script to the RLDS directory
cp pointlibero/LIBERO_Spatial_dataset_builder.py rlds_dataset_builder/rlds_dataset_builder_main/LIBERO_Spatial/

# 4. Run the builder
python LIBERO_Spatial_dataset_builder.py
```
Repeat the above steps on all suits to generate the complete PointLIBERO
