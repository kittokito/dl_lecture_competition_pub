import torch
import numpy as np
from scipy.fft import fft, fftfreq
import os
from scipy.signal import butter, filtfilt, sosfiltfilt

# データをdriveからロードする
data_train = torch.load('/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/dl_lecture_competition/dl_lecture_competition_pub/data/train_X.pt')
data_val = torch.load('/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/dl_lecture_competition/dl_lecture_competition_pub/data/val_X.pt')
data_test = torch.load('/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/dl_lecture_competition/dl_lecture_competition_pub/data/test_X.pt')

# ベースライン補正関数
def baseline_correction(data, baseline_period):
    baseline = data[:, :, :baseline_period].mean(axis=-1, keepdims=True)
    return data - baseline

# 前処理を適用する関数
def preprocess_data(file_path, baseline_period):
    data = torch.load(file_path).numpy()
    # ベースライン補正の適用
    corrected_data = baseline_correction(filtered_data, baseline_period)
    # Tensorに変換する
    corrected_data = torch.tensor(corrected_data, dtype=torch.float32)

    return corrected_data

baseline_period = 30  # ベースライン補正期間（サンプル数）

# 保存先の指定
data_dir = "/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/dl_lecture_competition/dl_lecture_competition_pub/data"
os.makedirs(data_dir, exist_ok=True)

# データの前処理の実行
preprocess_data(
    data_train,
    os.path.join(data_dir, 'train_X_filterd.pt'), 
    baseline_period )

preprocess_data(
    data_val,
    os.path.join(data_dir, 'val_X_filterd.pt'), 
    baseline_period )

preprocess_data(
    data_test,
    os.path.join(data_dir, 'test_X_filterd.pt'), 
    baseline_period )






