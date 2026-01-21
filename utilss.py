import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import os 
import torch.nn.functional as F
from configss import Configs
from rdkit import RDLogger 
import pickle 
from torch_geometric.utils import to_undirected 
# 新增 PDB 解析和 GVP 特征依赖
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial.distance import cdist 

# --- TDA 库导入和可用性检查 ---
try:
    from ripser import ripser
    from scipy.spatial.distance import pdist, squareform
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
# ------------------------

# 抑制 RDKit 警告
try:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR) 
except ImportError:
    pass

# --- 预处理文件目录常量 (不再是完整路径) ---
PREPROCESSED_DIR = 'preprocessed_data'

# --- 【新增】动态获取预处理文件路径函数 ---
def get_preprocessed_paths(config):
    """根据数据集名称动态生成预处理文件的完整路径"""
    dataset_name = config.dataset_name
    
    protein_path = os.path.join(
        PREPROCESSED_DIR, 
        f'protein_gvp_graphs_{dataset_name}.pkl' # 增加数据集名称
    )
    drug_path = os.path.join(
        PREPROCESSED_DIR, 
        f'drug_tda_features_{dataset_name}.pkl' # 增加数据集名称
    )
    return protein_path, drug_path

# --- 辅助函数：药物图结构特征提取 (保持不变) ---
def atom_features(atom):
    allowable_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    atomic_num = atom.GetAtomicNum()
    if atomic_num in [6, 7, 8, 16, 15, 9, 17, 35, 53]:
        feature = np.zeros(10)
        feature[allowable_set.index(atom.GetSymbol())] = 1
    else:
        feature = np.array([0] * 9 + [1]) 
    
    additional_features = np.array([
        atom.GetTotalNumHs(includeNeighbors=True), 
        atom.GetDegree(),                          
        atom.GetImplicitValence(),                 
        int(atom.GetIsAromatic()),                 
        atom.GetFormalCharge()                     
    ])
    
    # 总维度：10 + 5 = 15
    return np.concatenate([
        feature[:10], 
        additional_features
    ]).astype(np.float32) 

# --- 蛋白质 PDB 解析和 GVP 特征提取 (供 preprocess.py 调用) (保持不变) ---
AMINO_ACID_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

def get_protein_graph_features(pdb_file_path, cutoff=10.0):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file_path)
    except Exception:
        return None

    ca_atoms = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
                    residues.append(residue)
                    
    if not ca_atoms:
        return None

    # 1. 节点特征 (标量 x_s, 向量 x_v)
    coords = np.array([atom.get_coord() for atom in ca_atoms])
    N = coords.shape[0]
    
    x_s_list = []
    for res in residues:
        resname = res.get_resname()
        one_hot = np.zeros(len(AMINO_ACID_MAP), dtype=np.float32)
        if resname in AMINO_ACID_MAP:
            one_hot[AMINO_ACID_MAP[resname]] = 1.0
        x_s_list.append(one_hot)
        
    x_s = torch.tensor(np.array(x_s_list), dtype=torch.float)
    x_v = torch.zeros(coords.shape[0], 0, 3, dtype=torch.float) 

    # 2. 边 (Edge Index)
    ns = NeighborSearch(ca_atoms)
    edge_index = []
    
    for i in range(len(ca_atoms)):
        center_atom = ca_atoms[i]
        neighbors = ns.search(center_atom.get_coord(), cutoff, level='A')
        for neighbor_atom in neighbors:
            j = ca_atoms.index(neighbor_atom)
            if i != j:
                edge_index.append((i, j))
                
    if not edge_index:
        return None
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    # 3. 边特征 (Edge Attribute)
    row, col = edge_index
    edge_attr = torch.norm(torch.tensor(coords[row] - coords[col], dtype=torch.float), dim=1).unsqueeze(1)
    
    config = Configs() 
    # 注意：需要确保 Configs() 是在 CPU 上运行的，以防 temp_proj 默认跑到 GPU 上
    temp_proj = torch.nn.Linear(1, config.d_model)
    temp_proj.to('cpu') 
    edge_attr_proj = F.relu(temp_proj(edge_attr))
    
    data = Data(
        x_s=x_s, 
        x_v=x_v, 
        edge_index=edge_index, 
        edge_attr=edge_attr_proj,
        num_nodes=N
    )

    return data

def get_persistence_homology_features(pos, topo_dim):
    global TDA_AVAILABLE
    if not TDA_AVAILABLE:
        return torch.rand(topo_dim, dtype=torch.float) 

    try:
        pos_np = pos.cpu().numpy()
        
        # H2 计算通常至少需要 4 个点来构成空腔，如果点太少直接返回零向量
        if pos_np.shape[0] < 4:
             return torch.zeros(topo_dim, dtype=torch.float)

        # 1. 计算持久图，注意 maxdim=2 代表计算 H0, H1, H2
        diagrams = ripser(pos_np, maxdim=2)['dgms']
        
        feature_list = []
        
        # 2. 依次提取 H0, H1, H2 的特征
        for i in range(3):
            if i < len(diagrams) and diagrams[i].size > 0:
                dgm = diagrams[i]
                
                # 过滤掉包含 inf 的点 (H0 维度通常包含一个生命周期为无穷的成分)
                finite_mask = np.isfinite(dgm).all(axis=1)
                if np.any(finite_mask):
                    dgm_finite = dgm[finite_mask]
                    # 计算持久寿命: Death - Birth
                    persistence = dgm_finite[:, 1] - dgm_finite[:, 0]
                    feature_list.extend([np.mean(persistence), np.std(persistence)])
                else:
                    feature_list.extend([0.0, 0.0])
            else:
                # 如果该维度没有特征，填入 0
                feature_list.extend([0.0, 0.0])
        
        topo_features_np = np.array(feature_list, dtype=np.float32)
        
        # 3. 强制对齐维度，确保返回长度为 topo_dim (即 6)
        if len(topo_features_np) > topo_dim:
            topo_features_np = topo_features_np[:topo_dim]
        elif len(topo_features_np) < topo_dim:
            topo_features_np = np.pad(topo_features_np, (0, topo_dim - len(topo_features_np)))
            
        return torch.tensor(topo_features_np, dtype=torch.float)
        
    except Exception as e:
        # 出错时返回零向量
        return torch.zeros(topo_dim, dtype=torch.float)


# --- 数据集类 (保持不变) ---
class HGDDTIDataset(Dataset):
    # 构造函数现在接收预处理的图和TDA特征
    def __init__(self, df, esm_embeddings, protein_gvp_map, drug_tda_map, config): 
        self.df = df
        self.config = config
        self.esm_embeddings = esm_embeddings 
        self.protein_gvp_map = protein_gvp_map 
        self.drug_tda_map = drug_tda_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        drug_smiles = row['Drug']
        protein_sequence_key = row['Target Sequence'] 
        protein_id = row['Target_ID']
        
        try:
            affinity = float(row['Label']) 
        except ValueError:
            return None 
            
        # 1. 蛋白质 ESM 嵌入
        if protein_sequence_key not in self.esm_embeddings:
             return None 
             
        esm_seq_vec = torch.tensor(self.esm_embeddings[protein_sequence_key], dtype=torch.float)
        
        if esm_seq_vec.dim() == 2:
            protein_esm_vec = torch.mean(esm_seq_vec, dim=0) 
        elif esm_seq_vec.dim() == 1:
            protein_esm_vec = esm_seq_vec
        else:
            return None
            
        if protein_esm_vec.size(0) != self.config.protein_esm_dim:
             return None
        
        # 2. 蛋白质结构 (GVP 图特征) - 从 Map 中加载
        if protein_id not in self.protein_gvp_map:
            return None
        protein_data = self.protein_gvp_map[protein_id]
        
        # 3. 药物结构 (图特征 + 3D 坐标)
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None:
            return None 

        try:
            # 重新生成 3D 构象用于 pos 和 Drug Data 对象
            mol_h = Chem.AddHs(mol) 
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if mol_h.GetNumConformers() == 0:
                 AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3(), maxAttempts=50)
            if mol_h.GetNumConformers() == 0:
                return None 
            
            conf = mol_h.GetConformer()
            pos_tensor = torch.tensor(conf.GetPositions()[:mol.GetNumAtoms()], dtype=torch.float) 
        except Exception:
            return None 

        atom_f = []
        for atom in mol.GetAtoms():
            atom_f.append(atom_features(atom))
            
        if not atom_f:
             return None
             
        x_d = torch.tensor(np.array(atom_f), dtype=torch.float) 

        edge_index_d = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index_d.extend([(i, j), (j, i)])
        
        if not edge_index_d: 
             return None

        edge_index_d = torch.tensor(edge_index_d, dtype=torch.long).t().contiguous()
        edge_index_d = to_undirected(edge_index_d)
        
        # Drug Data 对象
        drug_data = Data(x=x_d, edge_index=edge_index_d, y=torch.tensor([affinity], dtype=torch.float),
                    pos=pos_tensor)
        
        # 4. 药物序列编码 (Morgan FP)
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.config.drug_fp_size)
        drug_token = torch.tensor([int(bit) for bit in mol_fp.ToBitString()], dtype=torch.float)
        
        # 5. 持久同调特征提取 - 从 Map 中加载
        if drug_smiles not in self.drug_tda_map:
             return None
        topo_features = self.drug_tda_map[drug_smiles]

        # 返回值顺序: drug_data, protein_data, drug_token, protein_esm_vec, affinity, topo_features
        return drug_data, protein_data, drug_token, protein_esm_vec, affinity, topo_features

# --- 数据加载与K折分割函数 (修改以加载预处理文件) ---
def load_data(config):
    """加载数据，同时加载 ESM 嵌入、GVP图和TDA特征。"""
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"数据文件未找到: {config.data_path}")
    
    df = pd.read_csv(config.data_path)
    
    if 'Label' not in df.columns:
         raise KeyError("数据文件中必须包含 'Label' 列用于分类任务，但未找到。")
         
    if 'Target_ID' not in df.columns:
         raise KeyError("数据文件中必须包含 'Target_ID' 列用于加载 PDB 文件，但未找到。")
         
    # 1. 加载 ESM 嵌入
    try:
        with open(config.esm_embedding_path, 'rb') as f:
            esm_embeddings = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ESM 嵌入文件未找到: {config.esm_embedding_path}")
    except Exception as e:
        raise Exception(f"加载 ESM 嵌入文件时出错: {e}")
        
    # 【修改点 1】: 获取动态路径
    PROTEIN_GVP_PATH, DRUG_TDA_PATH = get_preprocessed_paths(config)

    # 2. 加载 GVP 图 (预处理)
    if not os.path.exists(PROTEIN_GVP_PATH):
        raise FileNotFoundError(f"请先运行 preprocess.py! GVP 图文件未找到: {PROTEIN_GVP_PATH}")
    try:
        with open(PROTEIN_GVP_PATH, 'rb') as f:
            protein_gvp_map = pickle.load(f)
    except Exception as e:
        raise Exception(f"加载 GVP 图文件时出错: {e}")
        
    # 3. 加载 TDA 特征 (预处理)
    if not os.path.exists(DRUG_TDA_PATH):
        raise FileNotFoundError(f"请先运行 preprocess.py! TDA 特征文件未找到: {DRUG_TDA_PATH}")
    try:
        with open(DRUG_TDA_PATH, 'rb') as f:
            drug_tda_map = pickle.load(f)
    except Exception as e:
        raise Exception(f"加载 TDA 特征文件时出错: {e}")
        
    # 返回所有四个组件
    return df, esm_embeddings, protein_gvp_map, drug_tda_map

def get_k_fold_data(df, n_splits, random_state):
    """根据标签进行分层K折交叉验证，返回训练集和测试集的数据框列表。"""
    
    if 'Label' not in df.columns:
        raise ValueError("DataFrame 必须包含 'Label' 列进行分层抽样。")
        
    labels = df['Label'].values
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_index, test_index in kf.split(df, labels): 
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        folds.append((train_df, test_df))
        
    return folds

# --- Collate 函数 (保持不变) ---
def collate_fn_combined(batch):
    """处理药物图、蛋白质图、序列和拓扑特征的批处理。"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None, None, None, None

    # 拆分数据 (注意顺序: 药物图, 蛋白质图, 药物序列, 蛋白质ESM, Affinity, TDA)
    drug_data_list = [item[0] for item in batch]
    protein_data_list = [item[1] for item in batch] # GVP 图
    drug_token_list = [item[2] for item in batch]
    protein_esm_vec_list = [item[3] for item in batch] # ESM 序列
    affinity_list = [item[4] for item in batch]
    topo_feature_list = [item[5] for item in batch]

    # 药物图批处理
    drug_graph_batch = Batch.from_data_list(drug_data_list)
    
    # 蛋白质图批处理 (GVP 特征)
    protein_batch_data = Batch.from_data_list(protein_data_list)
    protein_graph_batch = Data(
        x_s=protein_batch_data.x_s,
        x_v=protein_batch_data.x_v,
        edge_index=protein_batch_data.edge_index,
        edge_attr=protein_batch_data.edge_attr,
        batch=protein_batch_data.batch # 必须包含 batch 索引
    )
    
    drug_seq_batch = torch.stack(drug_token_list, dim=0) 
    protein_esm_batch = torch.stack(protein_esm_vec_list, dim=0) # ESM 批处理
    affinity_batch = torch.tensor(affinity_list, dtype=torch.float).unsqueeze(1)
    topo_batch = torch.stack(topo_feature_list, dim=0) 
    
    # 返回值顺序: drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, affinity_batch, topo_batch
    return drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, affinity_batch, topo_batch