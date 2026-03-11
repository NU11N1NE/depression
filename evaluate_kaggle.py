import os
import sys
import torch

os.chdir("/kaggle/working/MSGAF")
os.environ["WANDB_MODE"] = "offline"
torch.backends.cudnn.benchmark = True

# 评估参数
sys.argv = [
    "evaluate.py",
    "--eval_config", "configs/eval_configs/eval_daicwoz_test_config.yaml",
    "--output_dir", "/kaggle/working/daicwoz-eval-results",  # 评估结果保存到工作目录
    "--checkpoint_kind", "best",
    "--name", "daicwoz-kaggle-run",
    "--n_temporal_windows", "1",
    "--seconds_per_window", "6",
    "--batch_size", "4",
    "--group", "baseline-daicwoz-kaggle",
    "--env", "reading-between-the-frames",
    "--use_modalities", "daic_audio_mfcc", "daic_audio_egemaps", "daic_facial_aus", "daic_gaze", "daic_head_pose", "daic_text"
]

# 执行评估
exec(open("evaluate.py").read())