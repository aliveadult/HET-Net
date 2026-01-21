import os
import torch
import numpy as np

class Configs:
    def __init__(self):
        # --- 通用设置 ---
        # 请根据你的实际路径修改 data_path
        self.data_path = '/media/6t/hanghuaibin/SaeGraphDTIII/data/DAVIS/dataset.csv' 
        self.output_dir = 'output/hgddti_gvp_esm/' # 新的输出目录
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # ！！！内存优化与配置修复！！！
        self.n_splits = 5             # K-Fold 交叉验证折数
        self.batch_size = 256           # 内存优化：批处理大小
        self.epochs = 300             # 训练轮次
        self.lr = 1e-3               # 学习率
        self.weight_decay = 1e-4      # 权重衰减
        self.random_state = 42 
        
        # 二分类阈值
        self.affinity_threshold = 7.0
        
        # --- ESM 嵌入设置 (保留) ---
        self.esm_embedding_path = '/media/6t/hanghuaibin/SaeGraphDTIII/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        
        # --- 蛋白质结构特征设置 (GVP) ---
        self.pdb_structure_path = '/media/6t/hanghuaibin/SaeGraphDTIII/DAVIS_predicted_structures'
        self.num_gvp_layers = 4       # GVP Block 层数
        self.gvp_h_dim = (100, 16)    # GVP 隐藏层标量/向量特征维度
        self.gvp_o_dim = (200, 32)    # GVP 输出标量/向量特征维度
        
        # --- 拓扑特征设置 ---
        self.topo_feature_dim = 6     # 持久同调特征的维度（B1: 均值和标准差）

        # --- 序列 Transformer 参数 ---
        self.d_model = 256          # 特征维度 (用于序列投影和药物GAT)
        self.nhead = 8
        self.num_transformer_layers = 6
        self.dropout = 0.4
        
        # 药物特征
        self.drug_fp_size = 1024       
        self.drug_vocab_size = self.drug_fp_size      
        self.drug_seq_len = self.drug_fp_size 
        
        # --- 图结构参数 ---
        self.num_diffusion_steps = 6 
        self.num_heads_gat = 8       
        
        self.drug_node_dim = 15       
        
        # ！！！关键修复：GVP 编码器输入的标量和向量维度！！！
        self.protein_node_s_dim = 20  # 蛋白质残基 one-hot 编码维度
        self.protein_node_v_dim = 0   # 蛋白质残基向量维度 
        
        # ！！！关键修改！！！更新融合维度：
        self.fusion_dim = self.d_model * 4 + self.gvp_o_dim[0]
        
        # --- 【新增】数据集名称，用于预处理文件命名 ---
        self.dataset_name = self._get_dataset_name()


    def _get_dataset_name(self):
        """从 data_path 中提取数据集名称（例如：'KIBA_dataset'）"""
        # 提取文件名 (dataset.csv)
        base_name = os.path.basename(self.data_path)
        # 提取上一级目录名 (KIBA)
        parent_dir = os.path.basename(os.path.dirname(self.data_path))
        
        # 组合成一个唯一的标识符，去除 .csv 后缀
        name = f"{parent_dir}_{os.path.splitext(base_name)[0]}"
        
        # 确保名称是有效的，如果路径是 /dataset.csv 这种形式，则只取 base_name
        if name.startswith('_'):
             return os.path.splitext(base_name)[0]
             
        return name.replace('.', '_')