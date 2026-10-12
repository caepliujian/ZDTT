import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "DataSets", "Property Dataset", "ThermalDecomposition")

INPUT_CSV = os.path.join(DATASET_DIR, "thermal_decomposition_full.csv")
TRAIN_OUTPUT_CSV = os.path.join(DATASET_DIR, "thermal_decomposition_train_random.csv")
TEST_OUTPUT_CSV = os.path.join(DATASET_DIR, "thermal_decomposition_test_random.csv")

SMILES_COL = "smiles"
TEST_RATIO = 0.2
RANDOM_SEED = 42
DROP_INVALID_SMILES = True


def is_valid_smiles(smiles):
    try:
        return isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def main():
    print(f"读取数据集: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

    if not 0 < TEST_RATIO < 1:
        raise ValueError("TEST_RATIO 必须在 0 和 1 之间。")

    df = pd.read_csv(INPUT_CSV)
    print(f"原始数据行数: {len(df)}")

    if SMILES_COL not in df.columns:
        raise ValueError(f"数据中找不到 SMILES 列: {SMILES_COL}")

    if DROP_INVALID_SMILES:
        valid_mask = df[SMILES_COL].apply(is_valid_smiles)
        invalid_count = int((~valid_mask).sum())
        if invalid_count > 0:
            print(f"检测到无效 SMILES: {invalid_count} 条，已在划分前剔除。")
        df = df[valid_mask].reset_index(drop=True)

    test_df = df.sample(frac=TEST_RATIO, random_state=RANDOM_SEED)
    train_df = df.drop(index=test_df.index)

    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df.to_csv(TRAIN_OUTPUT_CSV, index=False)
    test_df.to_csv(TEST_OUTPUT_CSV, index=False)

    print("随机划分完成。")
    print(f"训练集行数: {len(train_df)}")
    print(f"测试集行数: {len(test_df)}")
    print(f"训练集已保存至: {TRAIN_OUTPUT_CSV}")
    print(f"测试集已保存至: {TEST_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
