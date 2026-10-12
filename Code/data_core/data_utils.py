import pandas as pd
import numpy as np
from typing import List, Tuple
from chemprop.data import MoleculeDataset, MoleculeDatapoint
from chemprop.features import get_features_generator
from rdkit import Chem
from sklearn.model_selection import KFold

def load_data(path: str, smiles_col: str, target_col: str = None, 
              features_generator: List[str] = None) -> MoleculeDataset:
    """
    加载CSV数据并转换为Chemprop的数据集格式
    """
    df = pd.read_csv(path)
    
    # 确保列名存在
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES列 '{smiles_col}' 不在数据中")
    
    smiles_list = df[smiles_col].tolist()
    
    targets = None
    if target_col:
        if target_col not in df.columns:
             # 如果是预测模式，可能没有target列，或者target列名不对
             print(f"警告: 目标列 '{target_col}' 未找到，将作为预测模式处理")
        else:
            targets = df[[target_col]].values.tolist() # Chemprop expects list of lists for targets
    
    # 构建数据点
    datapoints = []
    for i, smiles in enumerate(smiles_list):
        target = targets[i] if targets else None
        datapoints.append(MoleculeDatapoint(smiles=[smiles], targets=target))
            
    dataset = MoleculeDataset(datapoints)
    
    # 如果需要生成特征 (如RDKit 2D描述符)
    if features_generator:
        # 注册并计算特征
        generators = []
        for gen_name in features_generator:
            # ================== 修复 Windows 下特征生成的崩溃 ==================
            # 原始的 chemprop 每次生成都会实例化新的 RDKit2DNormalized() 导致内存泄露和崩溃
            if gen_name == 'rdkit_2d_normalized':
                try:
                    from descriptastorus.descriptors import rdNormalizedDescriptors
                    # 全局缓存实例化对象
                    if not hasattr(get_features_generator, '_cached_rdkit_2d_normalized'):
                        get_features_generator._cached_rdkit_2d_normalized = rdNormalizedDescriptors.RDKit2DNormalized()
                    
                    def cached_rdkit_2d_normalized(mol):
                        try:
                            # 增加对空值或者处理失败的防御
                            with open('features_debug.log', 'a') as debug_f:
                                debug_f.write(f"    inside cached_rdkit_2d_normalized\n")
                                debug_f.flush()
                                smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
                                debug_f.write(f"    MolToSmiles done\n")
                                debug_f.flush()
                                res = get_features_generator._cached_rdkit_2d_normalized.process(smiles)
                                debug_f.write(f"    process done\n")
                                debug_f.flush()
                                if res is None:
                                    return [0.0] * 200
                                features = list(res[1:])
                                if len(features) != 200:
                                    return [0.0] * 200
                                # 修复 Windows numpy/stack 问题: 确保返回纯 python list float
                                return [float(x) if not pd.isna(x) and not np.isnan(x) and not np.isinf(x) else 0.0 for x in features]
                        except Exception as e:
                            with open('features_debug.log', 'a') as debug_f:
                                debug_f.write(f"    Exception in cached_rdkit_2d_normalized: {e}\n")
                                debug_f.flush()
                            return [0.0] * 200
                    
                    generators.append(cached_rdkit_2d_normalized)
                    print("已应用 rdkit_2d_normalized 的全局缓存优化，防止崩溃。")
                    continue
                except ImportError:
                    pass
            # =================================================================
            generators.append(get_features_generator(gen_name))
            
        # 批量计算特征 (并行化处理可能需要在Windows上小心)
        # 这里简单串行处理或者利用chemprop内部机制
        # Chemprop的dataset.reset_features_and_targets()通常会自动处理，但我们需要显式调用生成
        # 注意: Chemprop 1.6+ 推荐使用 get_data_from_smiles 或类似API，这里手动构建以便控制
        
        # 计算特征
        # 注意: 这里简化处理，直接调用features_generator
        # 实际项目中可能需要更复杂的并行计算
        import time
        from datetime import datetime
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating features (this may take a while)...")
        
        # 预先计算 fallback features size
        dummy_features = []
        for gen in generators:
            try:
                dummy_features.extend(gen(Chem.MolFromSmiles('C')))
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: generator failed on dummy molecule: {e}")
                
        fallback_len = len(dummy_features) if dummy_features else 200 # rdkit_2d_normalized usually has 200 features
        
        with open('features_debug.log', 'w') as log_f:
            for i, dp in enumerate(dataset):
                # 降低输出频率，避免命令行卡死
                if i > 0 and i % 1000 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating Features: {i}/{len(dataset)} processed...")
                log_f.write(f"[{i}/{len(dataset)}] Starting SMILES: {dp.smiles[0]}\n")
                log_f.flush()
                features = []
                try:
                    mol = Chem.MolFromSmiles(dp.smiles[0])
                    log_f.write(f"  mol created\n")
                    log_f.flush()
                    if mol:
                        for gen in generators:
                            log_f.write(f"  calling gen: {gen}\n")
                            log_f.flush()
                            feat = gen(mol)
                            log_f.write(f"  gen returned\n")
                            log_f.flush()
                            if feat is not None:
                                features.extend(feat)
                            log_f.write(f"  features set\n")
                            log_f.flush()
                    else:
                        log_f.write(f"  mol is None\n")
                        log_f.flush()
                        features = [0] * fallback_len
                except Exception as e:
                    # 捕获特征生成中的任何错误，避免整个程序崩溃
                    print(f"\nError generating features for SMILES {dp.smiles[0]}: {e}")
                    log_f.write(f"[{i}] Error: {e}\n")
                    log_f.write(f"  exception: {e}\n")
                    log_f.flush()
                    features = [0] * fallback_len
                    
                log_f.write(f"[{i}] Done.\n")
                log_f.flush()
                
                # 确保特征长度一致
                if len(features) == 0:
                    features = [0] * fallback_len
                
                dp.set_features(np.array(features))
            
    return dataset

def get_k_fold_indices(dataset_size: int, num_folds: int = 10, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    生成K-Fold交叉验证的索引
    返回: List of (train_indices, val_indices)
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    indices = np.arange(dataset_size)
    folds = []
    for train_idx, val_idx in kf.split(indices):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds

def split_data(dataset: MoleculeDataset, train_idx: List[int], val_idx: List[int]) -> Tuple[MoleculeDataset, MoleculeDataset]:
    """
    根据索引切分数据集
    """
    # Chemprop Dataset usually wraps a list. If it doesn't support list indexing, we do manual list comprehension.
    train_data = MoleculeDataset([dataset[i] for i in train_idx])
    val_data = MoleculeDataset([dataset[i] for i in val_idx])
    return train_data, val_data
