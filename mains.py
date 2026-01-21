import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from utilss import HGDDTIDataset, collate_fn_combined, load_data, get_k_fold_data
from models import HGDDTI
from configss import Configs
from evaluations import get_mean_and_std

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_classification_metrics(true_labels, pred_probs, threshold=0.5):
    """计算二分类指标：Acc, P, R, F1, AUC, AUPR"""
    
    pred_labels = (pred_probs >= threshold).astype(int)
    acc = accuracy_score(true_labels, pred_labels)
    
    if np.sum(true_labels) == 0 or np.all(true_labels == 1):
        p = precision_score(true_labels, pred_labels, zero_division=0)
        r = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        auc_score = 0.5 
        aupr_score = 0.0
    else:
        p = precision_score(true_labels, pred_labels, zero_division=0)
        r = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        auc_score = roc_auc_score(true_labels, pred_probs)
        aupr_score = average_precision_score(true_labels, pred_probs)
        
    return {
        'Acc': acc, 'P': p, 'R': r, 'F1': f1, 
        'AUC': auc_score, 'AUPR': aupr_score
    }

# --- Train 函数 (保持不变) ---
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, affinity_batch, topo_batch in tqdm(data_loader, desc="Training"):
        
        if drug_graph_batch is None:
            continue

        drug_graph_batch = drug_graph_batch.to(device)
        drug_seq_batch = drug_seq_batch.to(device)
        protein_esm_batch = protein_esm_batch.to(device)
        protein_graph_batch = protein_graph_batch.to(device) # GVP 图
        affinity_batch = affinity_batch.to(device)
        topo_batch = topo_batch.to(device) 
        
        optimizer.zero_grad()
        
        output_logits = model(drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, topo_batch)
        
        loss = criterion(output_logits, affinity_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

# --- Evaluate 函数 (保持不变) ---
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    true_labels = []
    pred_probs = []
    
    with torch.no_grad():
        for drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, affinity_batch, topo_batch in tqdm(data_loader, desc="Evaluating"):
            
            if drug_graph_batch is None:
                continue

            drug_graph_batch = drug_graph_batch.to(device)
            drug_seq_batch = drug_seq_batch.to(device)
            protein_esm_batch = protein_esm_batch.to(device)
            protein_graph_batch = protein_graph_batch.to(device) # GVP 图
            affinity_batch = affinity_batch.to(device)
            topo_batch = topo_batch.to(device) 

            output_logits = model(drug_graph_batch, drug_seq_batch, protein_esm_batch, protein_graph_batch, topo_batch)
            
            loss = criterion(output_logits, affinity_batch)
            total_loss += loss.item()
            
            probs = torch.sigmoid(output_logits).cpu().numpy().flatten()
            
            true_labels.extend(affinity_batch.cpu().numpy().flatten())
            pred_probs.extend(probs)

    avg_loss = total_loss / len(data_loader)
    
    metrics = calculate_classification_metrics(np.array(true_labels), np.array(pred_probs), threshold=0.5)
    metrics['Loss'] = avg_loss
    
    return metrics

def main():
    config = Configs()
    ensure_dir(config.output_dir)
    
    device = torch.device(config.device)
    
    # 关键修改：load_data 现在返回 df, esm_embeddings, protein_gvp_map, drug_tda_map
    try:
        df, esm_embeddings, protein_gvp_map, drug_tda_map = load_data(config)
    except FileNotFoundError as e:
        print(f"数据或预处理文件缺失: {e}")
        print("请确保已运行 python preprocess.py")
        return
    except Exception as e:
        print(f"数据加载失败，请检查文件权限或完整性: {e}")
        return

    k_folds = get_k_fold_data(df, config.n_splits, config.random_state)
    final_test_metrics = []
    
    drug_fp_size = config.drug_fp_size
    
    for fold, (train_df, test_df) in enumerate(k_folds):
        print(f"\n==================== Starting Fold {fold + 1}/{config.n_splits} ====================")
        
        model = HGDDTI(drug_fp_size, config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.BCEWithLogitsLoss() 
        
        # 关键修改：HGDDTIDataset 构造函数传入 GVP 和 TDA 映射
        train_dataset = HGDDTIDataset(train_df, esm_embeddings, protein_gvp_map, drug_tda_map, config)
        test_dataset = HGDDTIDataset(test_df, esm_embeddings, protein_gvp_map, drug_tda_map, config)

        train_loader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size, 
                                  shuffle=True, 
                                  collate_fn=collate_fn_combined, 
                                  num_workers=4)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config.batch_size, 
                                 shuffle=False, 
                                 collate_fn=collate_fn_combined, 
                                 num_workers=4)

        best_f1 = -1.0
        best_epoch = 0
        
        for epoch in range(1, config.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_metrics['Loss']:.4f} | "
                  f"Acc: {test_metrics['Acc']:.4f} | F1: {test_metrics['F1']:.4f} | "
                  f"AUC: {test_metrics['AUC']:.4f} | AUPR: {test_metrics['AUPR']:.4f}")

            if test_metrics['F1'] > best_f1:
                best_f1 = test_metrics['F1']
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(config.output_dir, f'best_model_fold_{fold+1}.pt'))
        
        model.load_state_dict(torch.load(os.path.join(config.output_dir, f'best_model_fold_{fold+1}.pt')))
        final_fold_metrics = evaluate(model, test_loader, criterion, device)
        final_test_metrics.append(final_fold_metrics)
        
        print(f"\nFold {fold + 1} Final Results (Epoch {best_epoch}): "
              f"Acc: {final_fold_metrics['Acc']:.4f}, F1: {final_fold_metrics['F1']:.4f}, "
              f"AUC: {final_fold_metrics['AUC']:.4f}, AUPR: {final_fold_metrics['AUPR']:.4f}")

    print("\n==================== K-Fold Cross-Validation Final Results ====================")
    metric_names = ['Acc', 'P', 'R', 'F1', 'AUC', 'AUPR']
    
    for name in metric_names:
        values = [m[name] for m in final_test_metrics]
        mean_val, std_val = get_mean_and_std(values)
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == '__main__':
    main()