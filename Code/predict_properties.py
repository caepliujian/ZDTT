import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# 修复 Windows 下由于 MKL/OpenMP 导致的静默崩溃和 FPE 异常
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Monkey-patch numpy to fix chemprop compatibility with numpy >= 1.25
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = DeprecationWarning
# 忽略所有的 numpy 浮点数警告，防止 MKL 捕获异常导致程序静默退出
np.seterr(all='ignore')

import rdkit
import descriptastorus

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader

from data_core.data_utils import load_data
from model_core.model_wrapper import DMPNNPredictor

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV (must contain "smiles" column)')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model.pt, scaler.pkl, config.yaml')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output CSV')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for prediction (overrides config if set)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from checkpoint dir
    config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Load Scaler
    scaler_path = os.path.join(args.checkpoint_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Load Model
    train_args = TrainArgs()
    train_args.depth = config['model']['depth']
    train_args.hidden_size = config['model']['hidden_size']
    train_args.ffn_num_layers = config['model']['ffn_num_layers']
    train_args.dropout = config['model']['dropout']
    train_args.undirected = config['model']['undirected']
    train_args.activation = config['model']['activation']
    train_args.features_generator = config['model']['features_generator']
    train_args.dataset_type = config['data']['dataset_type']
    train_args.device = device
    train_args.task_names = ["property"] # Dummy task name for prediction
    # 添加必要的缺失参数，避免 chemprop 内部报错
    train_args.features_scaling = False
    
    # Load Data first to get features_size
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    smiles_col = config['data']['smiles_col']
    if smiles_col not in df.columns:
        # Try finding 'smiles' case insensitive
        cols = [c.lower() for c in df.columns]
        if 'smiles' in cols:
            smiles_col = df.columns[cols.index('smiles')]
        else:
            raise ValueError(f"Input CSV must have '{smiles_col}' column")
            
    # Use load_data but target_col is None
    dataset = load_data(args.input, smiles_col=smiles_col, target_col=None, features_generator=config['model']['features_generator'])
    
    # Set features size
    if train_args.features_generator:
         if len(dataset) > 0 and dataset[0].features is not None:
             train_args.features_size = len(dataset[0].features)
         else:
             train_args.features_size = 0
    else:
         train_args.features_size = 0
         
    # Set default sizes if not provided (needed for FFN creation)
    if train_args.atom_descriptors_size is None:
        train_args.atom_descriptors_size = 0
    if train_args.atom_features_size is None:
        train_args.atom_features_size = 0
    if train_args.bond_features_size is None:
        train_args.bond_features_size = 0
    if train_args.ffn_hidden_size is None:
        train_args.ffn_hidden_size = train_args.hidden_size

    model = DMPNNPredictor(train_args)
    model.load(os.path.join(args.checkpoint_dir, 'model.pt'), device=device)
    model.to(device)
    model.eval()
    
    # 允许命令行覆盖 config 中的 batch_size，以加快推理速度
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    
    data_loader = MoleculeDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0, # Safer on Windows
        shuffle=False
    )
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Predicting...")
    preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            mol_batch = batch.batch_graph()
            features_batch = batch.features()
            
            batch_preds = model(mol_batch, features_batch)
            batch_preds = batch_preds.data.cpu().numpy()
            
            # Inverse transform
            batch_preds = scaler.inverse_transform(batch_preds)
            preds.extend(batch_preds.flatten().tolist())
            
    # Save results
    df['prediction'] = preds
    df.to_csv(args.output, index=False)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved predictions to {args.output}")

if __name__ == '__main__':
    # 将工作目录切换到脚本所在目录，避免相对路径报错
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
