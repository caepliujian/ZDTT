import os
import sys
import subprocess
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(SUBMISSION_DIR, "DataSets", "Property Dataset")
PREDICT_SCRIPT = os.path.join(SCRIPT_DIR, "predict_properties.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "metrics_only")

TASK_CONFIGS = [
    {
        "name": "DetonationVelocity",
        "data_path": os.path.join(DATASET_DIR, "DetonationVelocity", "detonation_velocity_test_fps.csv"),
        "target_col": "DetoD(m/s)",
        "checkpoint_dir": os.path.join(SCRIPT_DIR, "Checkpoints", "DetonationVelocity", "best_model"),
    },
    {
        "name": "ThermalDecomposition",
        "data_path": os.path.join(DATASET_DIR, "ThermalDecomposition", "thermal_decomposition_test_fps.csv"),
        "target_col": "Td",
        "checkpoint_dir": os.path.join(SCRIPT_DIR, "Checkpoints", "ThermalDecomposition", "best_model"),
    },
]

BATCH_SIZE = 1024
# 评测阶段优先用 CPU，通常更稳定；如果你确认环境稳定，可改成 "cuda"
PREFERRED_DEVICE = "cpu"
SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "Smiles"]


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def detect_smiles_column(columns):
    for candidate in SMILES_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    lowered = {str(column).lower(): column for column in columns}
    if "smiles" in lowered:
        return lowered["smiles"]

    raise ValueError("数据中找不到 SMILES 列。")


def run_prediction(input_csv, checkpoint_dir, output_csv):
    device_candidates = []
    preferred_device = PREFERRED_DEVICE.lower()
    if preferred_device == "cuda" and torch.cuda.is_available():
        device_candidates = ["cuda", "cpu"]
    else:
        device_candidates = ["cpu"]

    last_return_code = None
    for device in device_candidates:
        command = [
            sys.executable,
            PREDICT_SCRIPT,
            "--input",
            input_csv,
            "--checkpoint_dir",
            checkpoint_dir,
            "--output",
            output_csv,
            "--device",
            device,
            "--batch_size",
            str(BATCH_SIZE),
        ]

        log(f"开始预测: {os.path.basename(input_csv)} | device={device}")
        result = subprocess.run(command, cwd=SCRIPT_DIR)
        if result.returncode == 0:
            return

        last_return_code = result.returncode
        log(f"device={device} 预测失败，返回码: {result.returncode}")

    raise RuntimeError(f"预测失败，所有设备均尝试后仍失败，最后返回码: {last_return_code}")


def build_detailed_result(df, prediction_df, smiles_col, target_col):
    if "prediction" in prediction_df.columns:
        pred_series = prediction_df["prediction"]
    else:
        fallback_columns = [
            col for col in prediction_df.columns
            if col not in {smiles_col, target_col}
        ]
        if not fallback_columns:
            raise ValueError("预测输出中找不到 prediction 列。")
        pred_series = prediction_df[fallback_columns[-1]]

    result_df = df.copy()
    result_df["Predicted_Value"] = pd.to_numeric(pred_series, errors="coerce")
    result_df["True_Value"] = pd.to_numeric(result_df[target_col], errors="coerce")
    result_df["Error"] = result_df["Predicted_Value"] - result_df["True_Value"]
    result_df["Absolute_Error"] = result_df["Error"].abs()
    return result_df


def calculate_metrics(result_df):
    valid_df = result_df.dropna(subset=["True_Value", "Predicted_Value"]).copy()
    if valid_df.empty:
        raise ValueError("没有可用于评估的有效样本。")

    y_true = valid_df["True_Value"].to_numpy(dtype=float)
    y_pred = valid_df["Predicted_Value"].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "total_rows": len(result_df),
        "valid_rows": len(valid_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }, valid_df


def evaluate_one_task(task_config):
    name = task_config["name"]
    data_path = task_config["data_path"]
    target_col = task_config["target_col"]
    checkpoint_dir = task_config["checkpoint_dir"]

    log(f"开始评测任务: {name}")
    log(f"数据集路径: {data_path}")
    log(f"模型路径: {checkpoint_dir}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据集文件: {data_path}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"找不到模型目录: {checkpoint_dir}")

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"数据中找不到真实值列: {target_col}")

    smiles_col = detect_smiles_column(df.columns)

    temp_pred_path = os.path.join(OUTPUT_DIR, f"{name}_temp_predictions.csv")
    detailed_result_path = os.path.join(OUTPUT_DIR, f"{name}_detailed_predictions.csv")

    try:
        run_prediction(data_path, checkpoint_dir, temp_pred_path)
        prediction_df = pd.read_csv(temp_pred_path)
        result_df = build_detailed_result(df, prediction_df, smiles_col, target_col)
        metrics, valid_df = calculate_metrics(result_df)

        valid_df.to_csv(detailed_result_path, index=False)

        log(f"[{name}] 有效样本数: {metrics['valid_rows']} / {metrics['total_rows']}")
        log(f"[{name}] MAE: {metrics['mae']:.4f}")
        log(f"[{name}] RMSE: {metrics['rmse']:.4f}")
        log(f"[{name}] R²: {metrics['r2']:.4f}")
        log(f"[{name}] 详细结果已保存至: {detailed_result_path}")

        return {
            "Task": name,
            "Dataset_Path": data_path,
            "Checkpoint_Dir": checkpoint_dir,
            "Target_Column": target_col,
            "Total_Rows": metrics["total_rows"],
            "Valid_Rows": metrics["valid_rows"],
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
        }
    finally:
        if os.path.exists(temp_pred_path):
            os.remove(temp_pred_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_rows = []

    for task_config in TASK_CONFIGS:
        summary_rows.append(evaluate_one_task(task_config))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    log("全部评测完成。")
    log(f"汇总结果已保存至: {summary_path}")


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    main()
