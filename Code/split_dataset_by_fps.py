import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

def generate_fingerprints(smiles_list):
    """
    为SMILES列表生成Morgan指纹。
    """
    print("正在计算分子指纹...")
    fps = []
    valid_indices = []
    for i, smi in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            # 使用 Morgan 指纹 (半径为2，对应ECFP4，位数为2048)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_indices.append(i)
        else:
            print(f"警告: 第 {i} 行的SMILES无效: {smi}")
    return fps, valid_indices

def farthest_point_sampling(fps, sample_size, seed=42):
    """
    基于Tanimoto距离的最远点采样(FPS)。
    D(A, B) = 1 - S(A, B)
    """
    np.random.seed(seed)
    
    # 随机选择第一个点
    first_idx = np.random.randint(len(fps))
    selected_indices = [first_idx]
    
    # 初始化：所有点到已选集合(目前只有第一个点)的最短距离
    # DataStructs.BulkTanimotoSimilarity 返回的是相似度，我们用 1 减去它得到距离
    min_distances = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[first_idx], fps))
    
    print(f"开始进行最远点采样 (目标采样数量: {sample_size})...")
    for _ in tqdm(range(sample_size - 1)):
        # 1. 找到距离当前已选集合最远的点 (即最大化最小距离)
        new_idx = np.argmax(min_distances)
        selected_indices.append(new_idx)
        
        # 2. 计算所有点到这个新选出点的距离
        new_distances = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[new_idx], fps))
        
        # 3. 更新所有点到已选集合的最短距离
        min_distances = np.minimum(min_distances, new_distances)
        
    return selected_indices

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(script_dir), "DataSets", "Property Dataset", "ThermalDecomposition")
    input_path = os.path.join(dataset_dir, "thermal_decomposition_full.csv")
    train_output_path = os.path.join(dataset_dir, "thermal_decomposition_train_fps.csv")
    test_output_path = os.path.join(dataset_dir, "thermal_decomposition_test_fps.csv")
    
    # 测试集比例 (例如 0.2 表示 20% 作为测试集)
    test_ratio = 0.1
    
    print(f"加载数据: {input_path}")
    df = pd.read_csv(input_path)
    smiles_list = df['smiles'].tolist()
    
    # 1. 生成分子指纹并获取合法分子的索引
    fps, valid_indices = generate_fingerprints(smiles_list)
    
    # 过滤掉非法的行
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    num_total = len(valid_df)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_test
    
    print(f"总有效样本数: {num_total}")
    print(f"划分计划 -> 训练集: {num_train}, 测试集: {num_test}")
    
    # 2. 执行最远点采样 (FPS)
    # 我们用 FPS 来挑选测试集，这样挑选出来的测试集分子在化学空间中分布最广（代表性最强，能严格评估泛化能力）
    # 或者用 FPS 来挑选训练集，使训练集尽可能涵盖所有特征。
    # 通常的 Diversity Split 是用 FPS 挑选代表性子集，这里我们把选出来的作为独立测试集。
    test_indices_in_fps = farthest_point_sampling(fps, sample_size=num_test)
    
    # 3. 根据采样结果划分 DataFrame
    test_indices_set = set(test_indices_in_fps)
    train_indices_in_fps = [i for i in range(num_total) if i not in test_indices_set]
    
    test_df = valid_df.iloc[test_indices_in_fps].reset_index(drop=True)
    train_df = valid_df.iloc[train_indices_in_fps].reset_index(drop=True)
    
    # 4. 保存划分结果
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"划分完成！")
    print(f"训练集已保存至: {train_output_path} (形状: {train_df.shape})")
    print(f"测试集已保存至: {test_output_path} (形状: {test_df.shape})")

if __name__ == "__main__":
    main()
