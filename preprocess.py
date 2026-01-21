import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from configss import Configs
# 从 utilss 中导入必要的特征提取函数
from utilss import get_protein_graph_features, get_persistence_homology_features
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# 【新增导入】导入动态路径获取函数
from utilss import get_preprocessed_paths 

# 抑制 RDKit 警告
try:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR) 
except ImportError:
    pass

# --- 配置预处理输出目录常量 ---
PREPROCESSED_DIR = 'preprocessed_data'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_data(config):
    ensure_dir(PREPROCESSED_DIR)
    
    # 【修改点 1】: 获取动态路径
    PROTEIN_GVP_PATH, DRUG_TDA_PATH = get_preprocessed_paths(config)

    # 临时加载原始数据和 ESM 嵌入（此处是为了独立运行，避免循环依赖）
    print("1. 临时加载原始数据...")
    try:
        if not os.path.exists(config.data_path):
             raise FileNotFoundError(f"数据文件未找到: {config.data_path}")
        df = pd.read_csv(config.data_path)
            
    except Exception as e:
        print(f"数据加载失败，请检查 configss.py 路径配置: {e}")
        return

    # --- 2. 预计算蛋白质 GVP 图特征 ---
    print("\n2. 预计算蛋白质 GVP 图特征 (GVP_Structural_Features)...")
    protein_gvp_map = {}
    
    unique_protein_ids = df['Target_ID'].unique()
    
    for protein_id in tqdm(unique_protein_ids, desc="Processing Proteins"):
        pdb_file = os.path.join(config.pdb_structure_path, f'{protein_id}.pdb')
        
        if not os.path.exists(pdb_file):
            continue
            
        protein_data = get_protein_graph_features(pdb_file) 
        
        if protein_data is not None:
            # 确保张量是不可求导的
            protein_data.x_s = protein_data.x_s.detach()
            protein_data.x_v = protein_data.x_v.detach()
            protein_data.edge_index = protein_data.edge_index.detach()
            if protein_data.edge_attr is not None:
                 protein_data.edge_attr = protein_data.edge_attr.detach()
                 
            protein_gvp_map[protein_id] = protein_data 
            
    print(f"完成 GVP 图特征提取。有效蛋白质数量: {len(protein_gvp_map)}")

    # --- 3. 预计算药物 TDA 拓扑特征 ---
    print("\n3. 预计算药物 TDA 拓扑特征 (Topological_Features)...")
    drug_tda_map = {}
    
    unique_drug_smiles = df['Drug'].unique()
    temp_config = Configs() 
    
    for smiles in tqdm(unique_drug_smiles, desc="Processing Drugs"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        try:
            mol_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if mol_h.GetNumConformers() == 0:
                 AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3(), maxAttempts=50)
            if mol_h.GetNumConformers() == 0:
                continue 
                
            conf = mol_h.GetConformer()
            pos = conf.GetPositions()
            pos_tensor = torch.tensor(pos[:mol.GetNumAtoms()], dtype=torch.float) 

            topo_features = get_persistence_homology_features(
                pos_tensor, temp_config.topo_feature_dim
            )
            
            drug_tda_map[smiles] = topo_features.detach()
            
        except Exception:
            continue

    print(f"完成 TDA 特征提取。有效药物数量: {len(drug_tda_map)}")

    # --- 4. 保存预处理结果 ---
    print("\n4. 保存预处理文件...")
    with open(PROTEIN_GVP_PATH, 'wb') as f:
        pickle.dump(protein_gvp_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(DRUG_TDA_PATH, 'wb') as f:
        pickle.dump(drug_tda_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"所有预处理数据已保存到 {PREPROCESSED_DIR}/")

if __name__ == '__main__':
    config = Configs()
    try:
        from ripser import ripser
        TDA_AVAILABLE = True
    except ImportError:
        print("\n!!!!!!!! WARNING: ripser library not found. TDA features will be placeholders. !!!!!!!!\n")
        TDA_AVAILABLE = False
        
    try:
        from utilss import get_protein_graph_features, get_persistence_homology_features
        preprocess_data(config)
    except ImportError as e:
        print(f"请先确保 utilss.py 已更新，且必要的依赖库已安装。错误: {e}")