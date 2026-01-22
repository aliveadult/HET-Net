# HET-Net: Integrating Geometric Equivariance, Topological Data Analysis, and Multi-Scale Sequence Embeddings for Drug-Target Interaction Prediction

## 💡 HET-Net Framework

**HET-Net** (Heterogeneous Equivariant and Topological Network) is a novel deep learning framework designed to predict Drug-Target Interactions (DTI) with high fidelity. It addresses the limitations of existing models by incorporating fine-grained 3D geometry and intrinsic topological structures.

The framework integrates **five heterogeneous feature modalities**:

1. **Drug Sequence:** ECFP fingerprints processed via MLP.
2. **Drug Structure (Local):** 3D-Aware Graph Attention Network (GAT) incorporating explicit Euclidean distances.
3. **Drug Topology (Global):** Persistent Homology features (H0, H1, H2) extracted via Topological Data Analysis (TDA) to capture rings and cavities.
4. **Protein Sequence:** Evolutionary information from **ESM-2** embeddings.
5. **Protein Structure:** **Geometric Vector Perceptrons (GVP)** encoder ensuring E(3) translational and rotational equivariance.

## 🧠 File List

The project is structured as follows:

* **`mains.py`**: The entry point for the program. Handles the 5-Fold Cross-Validation loop, model initialization, training, and testing.
* **`models.py`**: Defines the core architecture, including:
* `HGDDTI`: The main fusion model (HET-Net).
* `GVP` & `ProteinStructuralEncoder`: Geometric Vector Perceptron layers for protein structure.
* `StructuralEncoder`: 3D-Aware GAT for drug structure.
* `TopologicalEncoder`: Network for processing TDA features.
* `DrugSequenceEncoder`: MLP for fingerprint processing.


* **`preprocess.py`**: **Crucial step.** Pre-calculates computationally expensive features:
* Generates GVP-ready graphs from Protein PDB files.
* Computes Persistent Homology (TDA) features for drugs using `ripser`.
* Saves results to the `preprocessed_data/` directory.


* **`configss.py`**: Central configuration file. Controls hyperparameters (batch size, learning rate), file paths (data, PDBs, embeddings), and model dimensions.
* **`utilss.py`**: Utility functions for:
* Loading datasets (`HGDDTIDataset`).
* Parsing PDB files using BioPython.
* Generating graph objects and TDA features.
* `collate_fn_combined`: Handling batches of heterogeneous data.


* **`evaluations.py`**: Contains metrics calculation (AUC, AUPR, F1, Accuracy, Precision, Recall).

## 🌐 Data Access
You can access the processed dataset files (including CSV labels, PDB structures, and ESM embeddings) via the following link:
> **Google Drive Path**: [https://drive.google.com/drive/folders/1U_bl2IDNV-FqyBD4tMJKbDQzEBiLYbin?hl=zh_CN](https://drive.google.com/drive/u/1/folders/1eXgtHrG6Uveyqj2zB0mrLg8OQlkDVSXj)

## 📁 Dataset & Preparation

### 1. Data Structure

To run the code, you need to configure `configss.py` to point to your data. The system expects:

* **CSV Dataset:** A file containing columns: Drug SMILES,Target Sequence, Target_ID, and Label(Binary 0/1).
* **Protein PDB Files:** A directory containing `.pdb` files named by their `Target_ID`.
* **ESM Embeddings:** A `.pkl` file containing pre-computed ESM-2 embeddings for the target sequences.

### 2. Supported Datasets

The model has been validated on standard benchmarks including:

* **Davis**
* **KIBA**
* **BindingDB**
* **BioSNAP**
* **C. elegans**

### 3. Preprocessing

**Before training**, you must run the preprocessing script to generate topological and graph features. This significantly speeds up training.

```bash
python preprocess.py

```

*This will create a `preprocessed_data` folder containing serialized graph and TDA features.*

## ✨ System Requirements

* **OS:** Linux (Recommended)
* **GPU:** CUDA-enabled GPU (Developed on NVIDIA RTX 4090)
* **Python:** 3.8+

## 🛠️ Environment Setup

You will need the following key libraries. Specifically, `ripser` is required for the Topological Data Analysis module.

```bash
# Create environment
conda create -n hetnet python=3.9
conda activate hetnet

# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Graph Neural Network dependencies
pip install torch_geometric

# Install Bioinformatics & TDA libraries
pip install rdkit
pip install biopython
pip install ripser  # Critical for TDA features
pip install scikit-learn
pip install tqdm
pip install pandas

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
✉ Citation
If you use HET-Net or this code in your research, please cite the following paper:

📧 Contact
Mingjian Jiang: jiangmingjian@qut.edu.cn

Huaibin Hang: School of Information and Control Engineering, Qingdao University of Technology
