import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DAIC-WOZ formant features",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="/kaggle/input/daic-woz-raw/data/")  # Kaggle原始数据路径
    parser.add_argument("--modality-id", type=str, default="audio_formant")
    parser.add_argument("--dest-root", type=str, default="/kaggle/working/daic-woz/no-chunked/")  # Kaggle工作目录
    args = parser.parse_args()

    featureID = "_FORMANT.csv"

    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessions = sorted(os.listdir(args.src_root))
    for sessionID in tqdm(sessions):
        # 跳过非目录文件
        if not os.path.isdir(os.path.join(args.src_root, sessionID)):
            continue

        data_path = os.path.join(args.src_root, sessionID, sessionID.split("_")[0] + featureID)
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping")
            continue

        df = pd.read_csv(data_path, header=None)
        seq = df.astype("float32").to_numpy()

        dest_path = os.path.join(dest_dir, sessionID + ".npz")
        np.savez_compressed(dest_path, data=seq)