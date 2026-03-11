import os
import sys
import torch

# 切换到仓库根目录
os.chdir("/kaggle/working/MSGAF")

# Kaggle环境配置
os.environ["WANDB_MODE"] = "offline"  # 关闭wandb，避免网络问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 便于调试CUDA错误
torch.backends.cudnn.benchmark = True  # 加速训练

# 训练参数配置
sys.argv = [
    "main.py",
    "--name", "daicwoz-kaggle-run",
    "--model_args.num_layers", "4",
    "--model_args.self_attn_num_heads", "2",
    "--model_args.self_attn_dim_head", "32",
    "--save_model", "1",
    "--n_temporal_windows", "1",
    "--seconds_per_window", "6",
    "--scheduler", "cosine",
    "--group", "baseline-daicwoz-kaggle",
    "--mode", "offline",
    "--epochs", "100",
    "--batch_size", "4",
    "--scheduler_args.max_lr", "0.0001",
    "--scheduler_args.end_epoch", "100",
    "--config_file", "configs/train_configs/baseline_daicwoz_config.yaml",
    "--dataset", "daic-woz",
    "--env", "reading-between-the-frames",
    "--use_modalities", "daic_audio_mfcc", "daic_audio_egemaps", "daic_facial_aus", "daic_gaze", "daic_head_pose", "daic_text",
    "--seed", "42"
]

# 执行训练
exec(open("main.py").read())