import argparse
import os
import pickle
import shutil
import sys
from typing import Dict

import numpy as np
import torch
import yaml

# 修复 Windows 下由于 MKL/OpenMP 导致的静默崩溃和 FPE 异常
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# 忽略所有的 numpy 浮点数警告，防止 MKL 捕获异常导致程序静默退出
np.seterr(all='ignore')

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataset, MoleculeDataLoader, StandardScaler
from chemprop.utils import makedirs

from model_core.model_wrapper import DMPNNPredictor
from training_core.engine import build_optimizer, evaluate, train_epoch
from data_core.data_utils import get_k_fold_indices, load_data, split_data

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_fold(fold_idx: int, train_data: MoleculeDataset, val_data: MoleculeDataset, config: Dict, device: torch.device):
    print(f"Running Fold {fold_idx}...")
    
    save_dir = os.path.join(config['training']['save_dir'], f'fold_{fold_idx}')
    makedirs(save_dir)
    
    # 1. Prepare data loaders
    # Scale targets
    scaler = StandardScaler().fit(train_data.targets())
    train_data.set_targets(scaler.transform(train_data.targets()).tolist())
    # Do NOT scale validation targets, so we can compare with raw values in evaluate()
    
    # Save scaler
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save config for reproduction
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    train_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=True
    )
    val_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=False
    )
    
    # 2. Initialize model
    # Convert config to TrainArgs for Chemprop compatibility
    args = TrainArgs()
    args.data_path = config['data']['train_data_path']
    args.dataset_type = config['data']['dataset_type']
    args.depth = config['model']['depth']
    args.hidden_size = config['model']['hidden_size']
    args.ffn_num_layers = config['model']['ffn_num_layers']
    args.dropout = config['model']['dropout']
    args.undirected = config['model']['undirected']
    args.activation = config['model']['activation']
    args.features_generator = config['model']['features_generator']
    args.device = device
    
    # Set task names and number of tasks
    args.task_names = ["property"] # Dummy task name
    # args.num_tasks = 1 # num_tasks is a property derived from task_names
    
    # Set features size
    if args.features_generator:
        # Get features size from first datapoint
        # Note: train_data is MoleculeDataset
        # We access the first datapoint's features
        sample_dp = train_data[0]
        if sample_dp.features is not None:
             args.features_size = len(sample_dp.features)
        else:
             args.features_size = 0
             print("WARNING: sample_dp.features is None!")
    else:
        args.features_size = 0
        
    # Set default sizes if not provided
    if args.atom_descriptors_size is None:
        args.atom_descriptors_size = 0
    if args.atom_features_size is None:
        args.atom_features_size = 0
    if args.bond_features_size is None:
        args.bond_features_size = 0
        
    # Set FFN hidden size same as MPNN hidden size if not specified
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    # Initialize model
    model = DMPNNPredictor(args)
    model.to(device)
    
    # 3. Optimizer and Scheduler
    # Need to set train_data_size in args or passed to build_optimizer
    # We pass it explicitly
    class OptArgs:
        pass
    opt_args = OptArgs()
    opt_args.init_lr = float(config['training']['init_lr'])
    opt_args.max_lr = float(config['training']['max_lr'])
    opt_args.final_lr = float(config['training']['final_lr'])
    opt_args.warmup_epochs = config['training']['warmup_epochs']
    opt_args.epochs = config['training']['epochs']
    opt_args.train_data_size = len(train_data)
    opt_args.batch_size = config['training']['batch_size']
    
    optimizer, scheduler = build_optimizer(model, opt_args)
    
    # 4. Training Loop
    loss_func = torch.nn.MSELoss() # Regression
    
    best_score = float('inf')
    best_model_path = os.path.join(save_dir, 'model.pt')
    
    for epoch in range(config['training']['epochs']):
        train_loss, _ = train_epoch(model, train_loader, optimizer, scheduler, loss_func, device, 0, args)
        
        # Evaluate
        # Note: evaluate() expects scaler to inverse transform predictions
        val_scores = evaluate(model, val_loader, scaler, device, metric_func=config['training']['metric'])
        val_score = val_scores[config['training']['metric']]
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val {config['training']['metric']} {val_score:.4f}")
        
        if val_score < best_score:
            best_score = val_score
            model.save(best_model_path)
    
    # Load best model and evaluate all metrics
    model.load(best_model_path, device)
    final_scores = evaluate(model, val_loader, scaler, device, metric_func=config['training']['metric'])
    print(f"Fold {fold_idx} Final Metrics: RMSE={final_scores['rmse']:.4f}, MAE={final_scores['mae']:.4f}, R2={final_scores['r2']:.4f}")
            
    # Cleanup: Restore original targets for next fold
    train_data.set_targets(scaler.inverse_transform(train_data.targets()).tolist())
            
    return best_score

def main():
    parser = argparse.ArgumentParser(description="DMPNN Training Script")
    parser.add_argument('--config', type=str, default='training_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    dataset = load_data(
        path=config['data']['train_data_path'],
        smiles_col=config['data']['smiles_col'],
        target_col=config['data']['target_col'],
        features_generator=config['model']['features_generator']
    )
    
    # Cross Validation
    num_folds = config['data']['num_folds']
    folds = get_k_fold_indices(len(dataset), num_folds=num_folds, seed=config['data']['seed'])
    
    fold_scores = []
    best_overall_score = float('inf')
    best_fold_idx = -1
    
    for i, (train_idx, val_idx) in enumerate (folds):
        train_data, val_data = split_data(dataset, train_idx, val_idx)
        score = run_fold(i, train_data, val_data, config, device)
        fold_scores.append(score)
        print(f"Fold {i} Best Score: {score:.4f}")
        
        if score < best_overall_score:
            best_overall_score = score
            best_fold_idx = i
        
    print(f"Mean {config['training']['metric']}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # ==========================================
    # 自动挑选并保存最佳模型
    # ==========================================
    print(f"\n==================================================")
    print(f"🌟 交叉验证完成！自动选出最佳模型: Fold {best_fold_idx} (Best Score: {best_overall_score:.4f})")
    
    best_fold_dir = os.path.join(config['training']['save_dir'], f'fold_{best_fold_idx}')
    best_model_dir = os.path.join(config['training']['save_dir'], 'best_model')
    
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
        
    shutil.copytree(best_fold_dir, best_model_dir)
    print(f"📁 最佳模型及其配置已完整复制到: {best_model_dir}")
    print(f"包含文件: {os.listdir(best_model_dir)}")
    print(f"==================================================\n")

if __name__ == '__main__':
    # 将工作目录切换到脚本所在目录，避免相对路径报错
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
