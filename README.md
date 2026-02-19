# MoCoV2 Contrastive Learning for Neuroimaging GWAS Pipeline

This repository contains the complete pipeline for training a contrastive learning model on T1 and T2 MRI images, extracting features for UK Biobank discovery cohort, performing PCA dimension reduction, and conducting GWAS analysis using FastGWA.

## Pipeline Overview

The pipeline consists of four main steps as illustrated in Figure 1 (`fig1.png`):

1. **Model Training**: Train a contrastive learning Vision Transformer (ViT) model on T1 and T2 MRI images
2. **Feature Extraction**: Extract features from the trained model for UK Biobank discovery cohort
3. **PCA Dimension Reduction**: Apply PCA to reduce feature dimensions
4. **GWAS Analysis**: Perform genome-wide association studies using FastGWA

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Model Training

Train the contrastive learning model using **paired T1 and T2 MRI volumes**.

#### Input

- **Manifest CSV**: A CSV file where each row is one subject. Two columns must contain paths to 3D MRI volumes:
  - One column for **T1** image paths
  - One column for **T2** image paths  
  Paths may be **absolute** or **relative**. Relative paths are resolved relative to the manifest CSV’s directory (so you can use filenames only if the volumes live in the same folder as the CSV). Column names are configurable via `--modality-t1` and `--modality-t2` (see below).
- **Volume format**: 3D neuroimaging volumes loadable with **NiBabel** (e.g. NIfTI `.nii`/`.nii.gz`, or MINC `.mnc` where supported). Each volume is normalized per-image (mean/std inside the brain mask) and padded as required by the model.

#### Input structure (CSV)

| Column (example names) | Content |
|------------------------|--------|
| `T1_unbiased_linear`   | Path to T1 volume for each subject (absolute or relative to manifest) |
| `T2_unbiased_linear`   | Path to T2 volume for each subject (absolute or relative to manifest) |

Example with **relative paths** (portable: place CSV and volumes in the same folder):

```csv
T1_unbiased_linear,T2_unbiased_linear
t1_icbm_normal_1mm_pn0_rf0.mnc,t2_icbm_normal_1mm_pn0_rf0.mnc
t1_icbm_normal_1mm_pn1_rf0.mnc,t2_icbm_normal_1mm_pn1_rf0.mnc
```

You can also use absolute paths. Column names can be overridden with `--modality-t1` and `--modality-t2`.

#### How to run

**Full training (with your own train/val CSV paths):**

Set `--output-dir` to the directory where you want checkpoints and TensorBoard logs saved (create it or let the script create it).

```bash
python engine_128_T1_T2_mocov2_contrast_learning.py \
    --train-datafile /path/to/train_manifest.csv \
    --val-datafile /path/to/val_manifest.csv \
    --output-dir /path/to/your/output_folder \
    [--modality-t1 T1_unbiased_linear] \
    [--modality-t2 T2_unbiased_linear] \
    [--resume /path/to/checkpoint.pth]
```

**Demo with phantom MRI images**

The phantom data is derived from simulated brain MRI (BrainWeb). https://brainweb.bic.mni.mcgill.ca/ **Cocosco, C.A., Kollokian, V., Kwan, R.K.-S., Evans, A.C.** *BrainWeb: Online Interface to a 3D MRI Simulated Brain Database.* NeuroImage, vol. 5, no. 4, part 2/4, S425, 1997.

1. **Download the phantom MRI data** from Google Drive and place the folder wherever you like (e.g. next to the repo):
   - **Download link**: [phantom_MRI (Google Drive)](https://drive.google.com/drive/folders/1AUO_2sAKgLI6UQyOfffdOQGlwwoTvkNv?usp=sharing)
   - The folder contains paired T1 and T2 MINC volumes and a **phantom_manifest.csv** that lists them using **relative paths** (filenames only). The training script resolves these paths relative to the manifest’s directory, so you do not need to edit the CSV when moving the folder.

2. **Set your output directory** and run training from the repo directory. Pass the **full path to the manifest CSV** and **`--output-dir`** where checkpoints and TensorBoard logs will be saved:

```bash
cd /path/to/mocov2_repo

python engine_128_T1_T2_mocov2_contrast_learning.py \
    --train-datafile /path/to/phantom_MRI/phantom_manifest.csv \
    --val-datafile /path/to/phantom_MRI/phantom_manifest.csv \
    --output-dir /path/to/your/output_folder
```

Use the same CSV for train and val for a small demo. Training runs for 300 epochs by default; use `--resume` to continue from a checkpoint.

**Rough timing:** With the small phantom dataset (10 pairs), 100 epochs on a single **NVIDIA H100** finish in about **15 minutes**. For real UK Biobank–scale data, expect roughly **19 hours** on **4× H100** GPUs. Adjust `--output-dir` and epoch count as needed.

**Anticipated results**

- **Small phantom dataset (10 pairs):** With very few samples, training loss does not decrease the way it does with large data and typically **stays around 3**. Linear CKA can reach about **0.9**, and the highest **retrieval recall@1** may go a bit **beyond 0.2**. This is expected for a tiny demo set.
- **Real UK Biobank–scale data:** With many subjects and many negative pairs, you can expect **loss to go under 0.1**, **linear CKA around 0.99**, and **retrieval recall@1 over 0.83**.

**Optional arguments:**

| Argument | Description |
|----------|-------------|
| `--train-datafile` | Path to CSV with T1/T2 paths for training |
| `--val-datafile` | Path to CSV for validation (defaults to train CSV if only `--train-datafile` is set) |
| `--output-dir` | Directory to save checkpoints and TensorBoard logs (default: `./output`); **set this to your desired output folder** |
| `--modality-t1` | Column name for T1 paths (default: `T1_unbiased_linear`) |
| `--modality-t2` | Column name for T2 paths (default: `T2_unbiased_linear`) |
| `--resume` | Path to checkpoint to resume training |
| `--start-epoch` | Epoch to start from (default: 0) |

This script trains a MoCoV2-based dual encoder model that learns representations from paired T1 and T2 MRI scans using contrastive learning.

**Key Features:**
- Vision Transformer (ViT) architecture
- Distributed training support
- Contrastive learning with momentum updates
- Saves model checkpoints during training

### Step 2: Feature Extraction for UK Biobank Discovery Cohort

Extract features from the trained model for the UK Biobank discovery cohort:

```bash
python extract_compute_pool_contrast_learning_saveT1.py \
    --checkpoint /path/to/trained/model/checkpoint.pth \
    --datafile /path/to/ukbiobank/data.csv \
    --output_dir /path/to/output/ \
    [--batch_size 32] \
    [--num_workers 4]
```

**Parameters:**
- `--checkpoint`: Path to the trained model checkpoint
- `--datafile`: Path to CSV file containing UK Biobank discovery cohort data
- `--output_dir`: Directory to save extracted features
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--num_workers`: Number of worker processes (default: 4)

This script extracts compute pool features from the trained model and saves them for downstream analysis. The pretained checkpoint could be found in the following link. https://drive.google.com/file/d/1VvmKhfLDk-JVpbYgHMu_djLvdA7SItMr/view?usp=sharing

### Step 3: PCA Dimension Reduction

Apply PCA to reduce the dimensionality of extracted features:

```bash
python apply_pca_to_features_T1.py \
    --checkpoint_path /path/to/inference_checkpoint.pt \
    --output_dir /path/to/output/ \
    [--n_components 128] \
    [--chunk_size 1000]
```

**Parameters:**
- `--checkpoint_path`: Path to the inference checkpoint containing extracted features
- `--output_dir`: Directory to save PCA-reduced features and models
- `--n_components`: Number of PCA components (default: 128)
- `--chunk_size`: Chunk size for processing large arrays (default: 1000)

This script performs incremental PCA on the extracted features to reduce dimensionality while preserving the most important variance.

### Step 4: GWAS Analysis with FastGWA

Perform genome-wide association studies on the PCA-reduced features using FastGWA:

```bash
python fastGWAS_csv.py [arguments]
```

This script runs FastGWA (fast genome-wide association analysis) on each PCA component as a phenotype, identifying genetic associations with the learned neuroimaging features.

**Output:**
- FastGWA results for each PCA component
- Association statistics and p-values
- Results can be used for downstream genetic analysis

## Pipeline Workflow

Refer to `fig1.png` for a visual representation of the complete pipeline workflow, showing the data flow from raw MRI images through model training, feature extraction, dimension reduction, and GWAS analysis.

## File Structure

- `engine_128_T1_T2_mocov2_contrast_learning.py`: Main training script for contrastive learning model
- `extract_compute_pool_contrast_learning_saveT1.py`: Feature extraction script for T1 images
- `apply_pca_to_features_T1.py`: PCA dimension reduction script
- `fastGWAS_csv.py`: FastGWA analysis script
- `fig1.png`: Pipeline workflow diagram

## Notes

- The pipeline is designed for distributed training and can utilize multiple GPUs
- Ensure sufficient disk space for storing extracted features and intermediate results
- The PCA step uses incremental PCA to handle large datasets efficiently
- FastGWA requires appropriate genetic data files (BGEN format) and sample files

## Citation

If you use this pipeline, please cite the relevant papers for:
- MoCoV2 contrastive learning framework
- Vision Transformer architecture
- FastGWA method

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

