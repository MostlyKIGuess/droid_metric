# droid_metric

<!--toc:start-->
- [droid_metric](#droidmetric)
  - [Installation](#installation)
    - [Option 1: Using Metric3D depth estimation (Python 3.9)](#option-1-using-metric3d-depth-estimation-python-39)
    - [Option 2: Using UniDepth depth estimation (Requires 2 conda environments)](#option-2-using-unidepth-depth-estimation-requires-2-conda-environments)
      - [Environment 1: UniDepth (Python 3.10+)](#environment-1-unidepth-python-310)
      - [Environment 2: DROID-SLAM (Python 3.9)](#environment-2-droid-slam-python-39)
  - [Usage](#usage)
    - [1. Setup pretrained models](#1-setup-pretrained-models)
    - [2. Utils](#2-utils)
    - [3. Run reconstruction](#3-run-reconstruction)
      - [Option A: Using Metric3D (Single environment workflow)](#option-a-using-metric3d-single-environment-workflow)
      - [Option B: Using UniDepth (Two-environment workflow)](#option-b-using-unidepth-two-environment-workflow)
    - [3*. Run reconstruction stepwise](#3-run-reconstruction-stepwise)
      - [Using Metric3D](#using-metric3d)
      - [Using UniDepth](#using-unidepth)
    - [4. Convert poses for evaluation](#4-convert-poses-for-evaluation)
    - [!Note](#note)
  - [Experiment](#experiment)
    - [1. Trajectory](#1-trajectory)
    - [2. Reconstruction](#2-reconstruction)
    - [3. Preview in the wild](#3-preview-in-the-wild)
<!--toc:end-->

This repo is for project combining [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [Metric3D](https://github.com/YvanYin/Metric3D), taking metric depth to improve the performance of DROID-SLAM in monocular mode.

Now also supports depths from [UniDepth](https://github.com/lpiccinelli-eth/UniDepth). Taking its depth.

## Installation

### Option 1: Using Metric3D depth estimation (Python 3.9)

```bash
# clone the repo with '--recursive' to get the submodules
# or run 'git submodule update --init --recursive' after cloning
cd droid_metric

# create conda env
conda create -n droid_metric python=3.9
conda activate droid_metric

# install requirements (other torch/cuda versions may also work)
pip install -r requirements.txt

# install droid-slam-backend
cd modules/droid_slam
python setup.py install
cd ../..
```

### Option 2: Using UniDepth depth estimation (Requires 2 conda environments)

**Important**: UniDepth requires Python 3.10+ due to its use of newer Python syntax (e.g., `int | list[int]` type hints), while the main DROID pipeline requires Python 3.9. Therefore, you need to set up two separate conda environments.

#### Environment 1: UniDepth (Python 3.10+)

```bash
# Create UniDepth environment
conda create -n unidepth python=3.11
conda activate unidepth

# Navigate to UniDepth module and install
cd modules/UniDepth
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

# Install optional dependencies for better performance
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

cd ../..
```

#### Environment 2: DROID-SLAM (Python 3.9)

```bash
# Create main DROID environment
conda create -n droid_metric python=3.9
conda activate droid_metric

# install requirements (other torch/cuda versions may also work)
pip install -r requirements.txt

# install droid-slam-backend
cd modules/droid_slam
python setup.py install
cd ../..

# IMPORTANT: Comment out UniDepth imports in modules/__init__.py
# Comment out these lines:
# from .unidepth import UniDepth
# "UniDepth": UniDepth,
```

*If you want to install specific version of `pytorch` and `cuda`, check [this link](https://pytorch.org/get-started/previous-versions/).*

*If you want to install `mmcv` under specific cuda version, check [this link](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).*

## Usage

### 1. Setup pretrained models

Download DROID-SLAM and Metric3D pretrained model running

```bash
python download_models.py
```

(Optional) Download ADVIO dataset running

```bash
python download_dataset.py
```

### 2. Utils

For camera calibration, check `scripts/calib.py`  
For video sampling, check `scripts/sample.py`

### 3. Run reconstruction

#### Option A: Using Metric3D (Single environment workflow)

```bash
python reconstruct.py --input ${RGB-images dir or video file} --output ${output dir} --intr ${intrinsic file} --viz
# for more options, check `reconstruct.py`
# if you do not provide intrinsic, it will be estimated as:
#  - fx = fy = max{image_width, image_height} * 1.2  (follow COLMAP)
#  - cx = image_width / 2
#  - cy = image_height / 2
```

#### Option B: Using UniDepth (Two-environment workflow)

**Step 1: Generate UniDepth depth maps**

```bash
# Switch to UniDepth environment
conda activate unidepth

# Generate depth maps using UniDepth
python depth_unidepth.py --images ${RGB-images dir} --out ${output dir} --intr ${intrinsic file}
# This script has the same arguments as depth.py but uses UniDepth models
```

**Step 2: Run DROID-SLAM with UniDepth depths**

```bash
# Switch to DROID environment
conda activate droid_metric

# Run DROID-SLAM using the generated UniDepth depths
python slam.py --images ${RGB-images dir} --depth ${depth data dir} --intr ${intrinsic file} --out-poses ${output poses dir} --viz

# Generate mesh reconstruction
python mesh.py --images ${RGB-images dir} --depth ${depth data dir} --poses ${poses dir} --intr ${intrinsic file} --save ${output mesh path}
```

### 3*. Run reconstruction stepwise

#### Using Metric3D

```bash
## depth estimate
python depth.py --images ${RGB-images dir} --out ${output dir} --intr ${intrinsic file}
# for more options, check `depth.py`

## droid-slam
python slam.py --images ${RGB-images dir} --depth ${depth data dir} --intr ${intrinsic file} --out-poses ${output poses dir} --viz
# for more options, check `slam.py`. You should run depth estimation first.

## mesh recon
python mesh.py --images ${RGB-images dir} --depth ${depth data dir} --poses ${poses dir} --intr ${intrinsic file} --save ${output mesh path}
# for more options, check `mesh.py`. You should run droid-slam first.
```

#### Using UniDepth

```bash
## depth estimate (in unidepth environment)
conda activate unidepth
python depth_unidepth.py --images ${RGB-images dir} --out ${output dir} --intr ${intrinsic file}

## droid-slam (in droid_metric environment)
conda activate droid_metric
python slam.py --images ${RGB-images dir} --depth ${depth data dir} --intr ${intrinsic file} --out-poses ${output poses dir} --viz

## mesh recon (in droid_metric environment)
python mesh.py --images ${RGB-images dir} --depth ${depth data dir} --poses ${poses dir} --intr ${intrinsic file} --save ${output mesh path}
```

### 4. Convert poses for evaluation

After running DROID-SLAM, you can convert the output poses to TUM format for evaluation with tools like EVO:

```bash
# Install EVO for trajectory evaluation
pip install evo

# Convert poses to TUM format
python convert_poses.py --input_dir path/to/poses --output_file trajectory.tum
```

The output TUM file can be used with EVO for trajectory evaluation and comparison:

```bash
# Example EVO usage
evo_traj tum trajectory.tum --plot
evo_ape tum ground_truth.tum trajectory.tum --plot
```

### !Note

The format of intrinsic file should be as follows (4 elements only):

```
# intrinsic.txt
${fx}
${fy}
${cx}
${cy}
```

**Important for UniDepth users**: Make sure to comment out the UniDepth imports in `modules/__init__.py` when working in the droid_metric environment:

```python
# Comment out these lines in modules/__init__.py:
# from .unidepth import UniDepth
# "UniDepth": UniDepth,
```

## Experiment

Tested on part of [ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) and [ADVIO](https://github.com/AaltoVision/ADVIO) dataset. `droid_D` refers to DROID-SLAM with Metric3D, `droid` refers to the original DROID-SLAM and `vslam` refers to the [OpenVSLAM](https://github.com/stella-cv/stella_vslam) framework. Notice that vslam method get lost on ICL-OfficeRoom-1 and all sequences of ADVIO.

### 1. Trajectory

![icl-traj](assets/traj_icl.png)

![advio-traj](assets/traj_advio.png)

*(some of the trajectories seem not aligned correctly, sorry for that.)*

### 2. Reconstruction

![mesh](assets/mesh.png)

### 3. Preview in the wild

![wild](assets/wild_p.png)
