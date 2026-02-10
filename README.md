
<p align="center">
  <h1 align="center">PointLIBERO: Unlocking Spatial Awareness in VLAs with a Novel 3D Dataset and a Lightweight framework</h1>
  <p align="center">


   <br />
    <strong>Meng Li</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    ·
    <a href="https://cv-shuchanglyu.github.io/EnHome.html"><strong>Shuchang Lyu</strong></a>
    ·
    <strong>Jun Jiang</strong></a>    
    ·
    <strong>Longhao Zou</strong></a>
    ·
    <a href="https://sites.google.com/view/guangliangcheng"><strong>Guangliang Cheng</strong></a>
    ·
    <br />
<p align="center">

    
  </p>





## Highlight!!!!
This repo is the implementation of "PointLIBERO: Unlocking Spatial Awareness in VLAs with a Novel 3D Dataset and a Lightweight framework".

## TODO
- [x] Release PointLIBERO
- [ ] Release CheckPoint and Test Code (1-2 months)
- [ ] Release Training Code (3-4 months)

##PointLIBERO Generation
* (1) Download the original [LIBERO](https://libero-project.github.io/main.html)
* (2) Install the LIBERO environment.
```
conda create -n  pointlibero python=3.10
conda activate
cd PointLibero
git clone Lifelong-Robot-Learning/LIBERO
pip install -e .
```
* (3) Select the LIBERO suit you want to generate.(eg.LIBERO Spatial)
```
cd PointLibero/experiments/robot/libero
python re_point_libero.py --libero_task_suit libero_spatial --libero_raw_data_dir the original path --libero_target_dir path to regenerated dataset directory

```
* (4) By running the code above, you will get an HDF5 file, but this is not the universal RLDS format.


