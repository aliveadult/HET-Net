import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool 
from torch_geometric.data import Batch

# --- GVP 核心模块 (保持不变) ---
class GVP(nn.Module):
    def __init__(self, s_dim, v_dim, s_out_dim, v_out_dim, vector_gate=True):
        super(GVP, self).__init__()
        self.vector_gate = vector_gate
        self.v_dim = v_dim
        
        # 标量 MLP
        self.s_proj = nn.Linear(s_dim + v_dim * 2, s_out_dim) # v_norms 和 v_norm_sq
        
        # 向量 MLP 
        if vector_gate:
            self.v_proj = nn.Linear(s_dim + v_dim * 2, v_out_dim * 2)
        else:
            self.v_proj = nn.Linear(s_dim + v_dim * 2, v_out_dim)
            
        # 向量特征维度投影 (Wx_v)
        self.v_out = nn.Linear(v_dim, v_out_dim, bias=False)

    def forward(self, h_s, h_v):
        ''' h_s: (N, s_dim), h_v: (N, v_dim, 3) '''
        
        # 1. 向量到标量的转换: ||h_v||, ||h_v||^2
        v_norms = torch.norm(h_v, dim=-1, p=2) # (N, v_dim)
        v_norm_sq = v_norms ** 2
        
        # 2. 拼接标量特征
        h = torch.cat([h_s, v_norms, v_norm_sq], dim=-1)
        
        # 3. 标量特征输出
        s_out = self.s_proj(h)

        # 4. 向量特征输出
        if self.vector_gate:
            gates = torch.sigmoid(self.v_proj(h)) 
            gate_s, gate_v = torch.chunk(gates, 2, dim=-1) 
            
            # Wx_v
            v_out_base = self.v_out(h_v.transpose(1, 2)).transpose(1, 2)
            
            # 门控: (N, v_out_dim, 1) * (N, v_out_dim, 3)
            v_out = v_out_base * gate_v.unsqueeze(-1)
        else:
            v_out = self.v_out(h_v.transpose(1, 2)).transpose(1, 2)
            
        return s_out, v_out

class GVPBlock(nn.Module):
    def __init__(self, h_dim, config):
        super(GVPBlock, self).__init__()
        
        s_dim, v_dim = h_dim
        s_out_dim, v_out_dim = h_dim # 保持输入输出维度一致
        
        # GVP 卷积层 (简化：只用于特征转换，不完全实现 GNN 聚合)
        self.gvp_conv = GVP(s_dim=s_dim, v_dim=v_dim, 
                            s_out_dim=s_out_dim, v_out_dim=v_out_dim)
        
        # 边特征处理，用于简化版 GVP-GNN 聚合
        self.edge_proj = nn.Linear(config.d_model, s_out_dim) 
        
        self.norm = nn.LayerNorm(s_out_dim)
        self.s_act = nn.SiLU()
        
    def forward(self, h_s, h_v, edge_index, edge_attr):
        
        # 1. 应用 GVP 变换
        h_s_out, h_v_out = self.gvp_conv(h_s, h_v)
        
        # 2. 简化的残差连接和激活
        h_s_res = F.relu(h_s + self.norm(h_s_out))
        h_v_res = h_v + h_v_out
        
        return h_s_res, h_v_res


# --- 蛋白质结构编码器 (基于 GVP) ---
class ProteinStructuralEncoder(nn.Module):
    def __init__(self, in_s_dim, in_v_dim, config):
        super(ProteinStructuralEncoder, self).__init__()
        
        # 1. 初始特征投影到 GVP 隐藏维度
        self.initial_proj = GVP(s_dim=in_s_dim, v_dim=in_v_dim, 
                                s_out_dim=config.gvp_h_dim[0], v_out_dim=config.gvp_h_dim[1])
        
        # 2. GVP 块堆叠
        blocks = []
        for i in range(config.num_gvp_layers):
            blocks.append(GVPBlock(config.gvp_h_dim, config)) 
        self.blocks = nn.ModuleList(blocks)
        
        # 3. 最终输出投影
        self.final_proj = GVP(s_dim=config.gvp_h_dim[0], v_dim=config.gvp_h_dim[1],
                              s_out_dim=config.gvp_o_dim[0], v_out_dim=config.gvp_o_dim[1])
        
        self.norm = nn.LayerNorm(config.gvp_o_dim[0])
        
    # 【关键】确保 forward 接收 x_s 和 x_v
    def forward(self, x_s, x_v, edge_index, edge_attr, batch_index):
        
        # 1. 初始投影
        # 如果 x_s 是 None，则此处会出错
        h_s, h_v = self.initial_proj(x_s, x_v)
        
        # 2. GVP 块堆叠
        for block in self.blocks:
            h_s, h_v = block(h_s, h_v, edge_index, edge_attr)
            
        # 3. 最终投影
        h_s_final, _ = self.final_proj(h_s, h_v) 
        
        # GVP 编码器返回经过 GVP 后的所有节点标量特征
        return self.norm(h_s_final) 


# --- 序列编码器 (药物) (保持不变) ---
class DrugSequenceEncoder(nn.Module):
    def __init__(self, fp_size, config):
        super(DrugSequenceEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(fp_size, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )

    def forward(self, seq_tokens):
        return self.proj(seq_tokens.float()) 

# --- 结构编码器 (3D 感知 GAT) (保持不变) ---
class StructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(StructuralEncoder, self).__init__()
        self.config = config
        self.dist_proj = nn.Linear(1, config.d_model) 
        self.conv1 = GATConv(in_dim, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        self.conv2 = GATConv(config.d_model, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        
    def forward(self, x, edge_index, pos):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1).unsqueeze(1) 
        dist_emb = F.relu(self.dist_proj(dist)) 
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        return x + h1 + h2 

# --- 拓扑特征编码器 (保持不变) ---
class TopologicalEncoder(nn.Module):
    def __init__(self, topo_feature_dim, config):
        super(TopologicalEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(topo_feature_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, topo_features):
        return self.proj(topo_features)

# --- 核心模型：HGDDTI (修改: 融合 ESM 和 GVP) ---
class HGDDTI(nn.Module):
    def __init__(self, drug_fp_size, config): 
        super(HGDDTI, self).__init__()
        self.config = config
        
        # 药物特征
        self.initial_atom_dim = 15 
        self.atom_proj = nn.Linear(self.initial_atom_dim, config.d_model)
        self.drug_seq_encoder = DrugSequenceEncoder(drug_fp_size, config)
        self.structural_encoder = StructuralEncoder(config.d_model, config)
        
        # 蛋白质特征 (ESM - 序列)
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, config.d_model) 
        
        # 蛋白质特征 (GVP - 结构)
        self.protein_structural_encoder = ProteinStructuralEncoder(
            in_s_dim=config.protein_node_s_dim, 
            in_v_dim=config.protein_node_v_dim, 
            config=config
        )
        
        # 药物拓扑特征
        self.topo_encoder = TopologicalEncoder(config.topo_feature_dim, config)
        
        self.fusion_dim = config.fusion_dim 
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, config.d_model), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1) 
        )

    # 【关键】确保 forward 接收 protein_esm_vecs 和 protein_graph_batch
    def forward(self, drug_graph_batch, drug_seq_data, protein_esm_vecs, protein_graph_batch, topo_features):
        
        # --- 1. 药物序列特征 ---
        drug_seq_vec = self.drug_seq_encoder(drug_seq_data)       
        
        # --- 2. 蛋白质序列特征 (ESM) ---
        protein_esm_vec = F.relu(self.protein_esm_proj(protein_esm_vecs)) 
        
        # --- 3. 药物结构特征 (GAT) ---
        x_d, edge_index_d, pos_d = drug_graph_batch.x, drug_graph_batch.edge_index, drug_graph_batch.pos 
        x_d = F.relu(self.atom_proj(x_d)) 
        structural_features_all = self.structural_encoder(x_d, edge_index_d, pos_d)
        drug_structural_vec = global_mean_pool(structural_features_all, drug_graph_batch.batch)
        
        # --- 4. 蛋白质结构特征 (GVP) ---
        # 【关键】正确解构 GVP 输入特征
        x_s_p, x_v_p, edge_index_p, edge_attr_p = (
            protein_graph_batch.x_s, protein_graph_batch.x_v, 
            protein_graph_batch.edge_index, protein_graph_batch.edge_attr
        ) 
        
        # 【关键】正确调用 encoder，传入 x_s_p 和 x_v_p
        protein_structural_features_all = self.protein_structural_encoder(
            x_s_p, x_v_p, edge_index_p, edge_attr_p, protein_graph_batch.batch
        )
        
        # 全局池化 (GVP 输出的标量特征)
        protein_structural_vec = global_mean_pool(
            protein_structural_features_all, 
            protein_graph_batch.batch
        )
        
        # --- 5. 拓扑特征编码 ---
        topo_vec = self.topo_encoder(topo_features)
        
        # --- 6. 特征融合与预测 ---
        fused_features = torch.cat([
            drug_seq_vec, 
            protein_esm_vec,       # 序列
            drug_structural_vec, 
            protein_structural_vec, # 结构
            topo_vec
        ], dim=1)
        
        output = self.fusion_head(fused_features)
        
        return output