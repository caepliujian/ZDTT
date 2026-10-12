import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import NoamLR
from tqdm import tqdm
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_epoch(
    model: nn.Module,
    data_loader: MoleculeDataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    loss_func: nn.Module,
    device: torch.device,
    n_iter: int = 0,
    args = None
) -> Tuple[float, int]:
    
    model.train()
    loss_sum = 0
    iter_count = 0
    
    for batch in tqdm(data_loader, desc="Training", leave=False):
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        targets = batch.targets()
        
        # Move to device handled by model usually, but Chemprop graphs handle device internally?
        # Actually Chemprop's BatchMolGraph handles device during forward pass if model is on device?
        # Wait, usually we pass graph to model.
        # Chemprop 1.6.1: model(batch_graph, features_batch)
        
        # Prepare targets
        targets = torch.tensor(targets, dtype=torch.float).to(device)
        mask = torch.tensor([[t is not None for t in x] for x in batch.targets()], dtype=torch.bool).to(device)
        targets = targets.masked_select(mask) # Flatten for regression
        
        # Forward pass
        if features_batch is None:
            print("WARNING: features_batch is None in train_epoch!")
        else:
            for i, f in enumerate(features_batch):
                if f is None:
                    print(f"WARNING: features_batch[{i}] is None in train_epoch!")
                    
        preds = model(mol_batch, features_batch)
        
        # Mask predictions
        preds = preds.masked_select(mask)
        
        if args and args.dataset_type == 'classification':
             # Handle classification logic if needed (sigmoid etc)
             pass
        
        loss = loss_func(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_sum += loss.item()
        iter_count += 1
        n_iter += 1
        
    return loss_sum / iter_count, n_iter

def evaluate(
    model: nn.Module,
    data_loader: MoleculeDataLoader,
    scaler: StandardScaler,
    device: torch.device,
    metric_func: str = 'rmse'
) -> Dict[str, float]:
    
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch: MoleculeDataset
            mol_batch = batch.batch_graph()
            features_batch = batch.features()
            
            batch_preds = model(mol_batch, features_batch)
            batch_targets = batch.targets()
            
            # Collect predictions and targets
            # Need to Inverse Scale predictions
            batch_preds = batch_preds.data.cpu().numpy()
            if scaler:
                batch_preds = scaler.inverse_transform(batch_preds)
            
            preds.extend(batch_preds.tolist())
            targets.extend(batch_targets)
            
    # Flatten lists (assuming single task regression for simplicity)
    # Filter out None targets
    valid_preds = []
    valid_targets = []
    for p, t in zip(preds, targets):
        # t is list of targets (multitask), here we assume single task or handle list
        if t[0] is not None:
            valid_preds.append(p[0])
            valid_targets.append(t[0])
            
    valid_preds = np.array(valid_preds)
    valid_targets = np.array(valid_targets)
    
    results = {}
    results['rmse'] = np.sqrt(mean_squared_error(valid_targets, valid_preds))
    results['mae'] = mean_absolute_error(valid_targets, valid_preds)
    results['r2'] = r2_score(valid_targets, valid_preds)
    
    return results

def build_optimizer(model: nn.Module, args) -> Tuple[Optimizer, _LRScheduler]:
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs],
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )
    return optimizer, scheduler
