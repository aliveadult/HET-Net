# HET-Net: Integrating Geometric Equivariance, Topological Data Analysis, and Multi-Scale Sequence Embeddings for Drug-Target Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

## 💡 HET-Net Framework

**HET-Net** (Heterogeneous Equivariant and Topological Network) is a cutting-edge deep learning framework for Drug-Target Interaction (DTI) prediction. Unlike traditional models that rely on simple sequence or 2D graph representations, HET-Net bridges the gap between microscopic atomic geometry and macroscopic molecular topology.



### Core Innovations
The framework integrates **five heterogeneous feature modalities** to capture chemically plausible interactions:

1.  **Drug Sequence ($F_{d-seq}$):** Extracts local chemical substructures using ECFP fingerprints (radius 2, 1024-bit) processed via a Multi-Layer Perceptron (MLP).
2.  **Drug Structure ($F_{d-struct}$):** A **3D-Aware Graph Attention Network (GAT)** that explicitly incorporates Euclidean distances between atoms to capture fine-grained local geometry.
3.  **Drug Topology ($F_{d-Topo}$):** Utilizes **Topological Data Analysis (TDA)** via Persistent Homology. It computes persistence diagrams for dimensions $H_0$ (connectivity), $H_1$ (loops/rings), and $H_2$ (voids/cavities) to characterize global molecular shape and stability.
4.  **Protein Sequence ($F_{p-seq}$):** Leverages the **ESM-2** evolutionary scale language model to generate rich, context-aware sequence embeddings.
5.  **Protein Structure ($F_{p-struct}$):** Implements a **Geometric Vector Perceptron (GVP)** encoder. This ensures $E(3)$ equivariance (invariance to translation and rotation), allowing the model to rigorously learn the 3D binding pocket geometry.

---

## 🧠 File Structure

The project code is modularized for clarity and reproducibility:

* **`mains.py`**
    * **Function:** The central entry point.
    * **Logic:** Orchestrates the 5-Fold Cross-Validation loop, initializes the `HGDDTI` model, manages the optimizer/scheduler, and executes training/testing pipelines.
* **`models.py`**
    * **Function:** Defines the neural network architectures.
    * **Components:**
        * `HGDDTI`: The main fusion network combining all 5 branches.
        * `GVP` & `ProteinStructuralEncoder`: $E(3)$-equivariant layers for processing protein 3D coordinates.
        * `StructuralEncoder`: 3D-Aware GAT for drug graphs.
        * `TopologicalEncoder`: MLP for processing TDA persistence statistics.
* **`preprocess.py`**
    * **Function:** Handles heavy computational tasks offline to speed up training.
    * **Tasks:**
        * Constructs protein contact graphs from `.pdb` files.
        * Computes Persistent Homology (TDA) features using the `ripser` library.
        * Saves serialized data dictionaries to `preprocessed_data/`.
* **`configss.py`**
    * **Function:** Global configuration hub.
    * **Settings:** Paths (datasets, PDBs, embeddings), hyperparameters (Batch size: 256, LR: 1e-3, Epochs: 300), and feature dimensions (GVP hidden dims, Attention heads).
* **`utilss.py`**
    * **Function:** Data handling and auxiliary tools.
    * **Tools:** `HGDDTIDataset` (PyTorch Dataset class), PDB parsing (BioPython), graph conversion tools, and `collate_fn_combined` for batching heterogeneous data.
* **`evaluations.py`**
    * **Function:** Performance metrics.
    * **Metrics:** AUC, AUPR, F1 Score, Accuracy, Precision, Recall, and MCC.

---

## 🌐 Data Access
You can access the processed dataset files (including CSV labels, PDB structures, and ESM embeddings) via the following link:
> **Google Drive Path**: [https://drive.google.com/drive/folders/1U_bl2IDNV-FqyBD4tMJKbDQzEBiLYbin?hl=zh_CN](https://drive.google.com/drive/u/1/folders/1eXgtHrG6Uveyqj2zB0mrLg8OQlkDVSXj)

## 📁 Datasets

This study utilizes **seven** authoritative benchmark datasets to validate model performance and generalization.

| Dataset | Type | Interaction Pairs | Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Davis** | Kinase | 25,772 | Measures $K_d$ values for kinase inhibitors. High-quality experimental data. | [Link](https://davischallenge.org/) |
| **KIBA** | Kinase | 118,254 | Combines $K_i$, $K_d$, and $IC_{50}$ into a single KIBA score. Large-scale integration. | [Link](https://paperswithcode.com/dataset/kiba) |
| **BindingDB** | General | 60,780 | A public database of measured binding affinities, focusing on protein-ligand complexes. | [Link](https://www.bindingdb.org/bind/) |
| **BioSNAP** | Interactions | 27,482 | Diverse biomedical network dataset from Stanford. | [Link](http://snap.stanford.edu/biodata) |
| **C. elegans** | Species | 7,511 | Specific interactions for the model organism *C. elegans*. | [Link](https://downloads.thebiogrid.org/Celegans_DTI/) |
| **Enzyme (E)** | Protein Family | 5,840 | Focused dataset for enzyme-ligand interactions. | [Reference](https://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) |
| **Ion Channel (IC)**| Protein Family | 2,950 | Focused dataset for ion channel modulators. | [Reference](https://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) |

### Data Preparation
To run the model, organize your data as follows:
1.  **CSV File:** Must contain columns `Drug` (SMILES), `Target Sequence`, `Target_ID`, and `Label` (0/1).
2.  **PDB Directory:** A folder containing `.pdb` structure files named strictly by their `Target_ID` (e.g., `1a2b.pdb`).
3.  **ESM Embeddings:** A `.pkl` file with pre-computed ESM-2 embeddings for all target sequences in the dataset.

---

## ✨ System Requirements & Installation

**Hardware:**
* **GPU:** CUDA-enabled NVIDIA GPU (Tested on dual RTX 4090 24G).
* **OS:** Linux (Recommended).

**Dependencies:**
The framework relies on `ripser` for topological calculations and `torch_geometric` for graph operations.

```bash
# 1. Create Conda Environment
conda create -n hetnet python=3.9
conda activate hetnet

# 2. Install PyTorch (Ensure CUDA version matches your system, e.g., 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install Graph Neural Network Libraries
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.1.0+cu121.html](https://data.pyg.org/whl/torch-2.1.0+cu121.html)

# 4. Install TDA and Bio-computation Libraries
pip install ripser          # Critical for Persistent Homology
pip install biopython       # For PDB parsing
pip install rdkit           # For molecular processing

# 5. Install General Utilities
pip install pandas tqdm scikit-learn fair-esm

```

*Note: If you need to generate ESM embeddings from scratch, install `fair-esm`, otherwise ensure you have the pre-computed `.pkl` file referenced in `configss.py`.*

## 🖥️ Run Code

1. **Configure:** Edit `configss.py` to set your `data_path`, `pdb_structure_path`, and `esm_embedding_path`.
2. **Preprocess:** Run feature extraction.
```bash
python preprocess.py

```


3. **Train:** Run the main training loop (performs 5-fold CV).
```bash
python mains.py

```



## ✉ Citation

If you use this code or framework in your research, please cite the following paper:

```bibtex
@article{
 
}

```

