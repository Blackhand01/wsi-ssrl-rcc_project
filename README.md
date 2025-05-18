# wsi-ssrl-rcc_project

This repository contains the code and experiments for classifying Renal Cell Carcinoma subtypes using self-supervised learning techniques on Whole Slide Images (WSIs). Project for MLA Course.

## Team Members

- **Stefano Roy Bisignano** 
- **Mirko Di Maggio** 
- **Rida Ben Hassi** 
---

## 🛠️ Prerequisites

1. **Git**  
2. **Miniconda** (or Anaconda)  
   - **macOS (Intel/ARM) & Linux**  
     ```bash
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
     chmod +x Miniconda3-*.sh
     ./Miniconda3-*.sh
     source ~/.bashrc  # or ~/.zshrc
     ```
   - Verify:
     ```bash
     conda --version
     ```
3. *(Optional)* **Google Drive for Desktop** in streaming mode (setting by Preferences option), if you prefer working with a local IDE (VSCode, PyCharm, etc.)

---

## 🚀 Clone the repository

```bash
git clone https://github.com/<your-org>/wsi-ssrl-rcc_project.git
cd wsi-ssrl-rcc_project
````

---

## 🔧 Create and activate the Conda environment

```bash
# (Re)move any existing env
conda env remove -n wsi-ssrl || true

# Create from spec
conda env create -f environment.yml

# Activate
conda activate wsi-ssrl
```

---

## 🔎 Useful commands

* Launch JupyterLab: `jupyter lab`
* Remove environment:

  ```bash
  conda deactivate
  conda env remove -n wsi-ssrl
  ```

* Convert python to notebook Jupyter: 
```bash
jupytext /Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/notebooks/1_prepare_metadata.py --to notebook
  ```
