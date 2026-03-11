import os
import torch
import yaml
from easydict import EasyDict
import numpy as np

# 配置路径
os.chdir("/kaggle/working/MSGAF")
CHECKPOINT_PATH = "/kaggle/working/checkpoints/baseline-daicwoz-kaggle-daicwoz-kaggle-run/best/epoch=xx-TemporalEvaluator_f1=xx.ckpt"  # 替换为实际最佳检查点路径
ONNX_SAVE_PATH = "/kaggle/working/daicwoz_depression_model.onnx"

# 加载配置
def load_config(config_path):
    with open(config_path, 'rt') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg

# 初始化模型
args = EasyDict({
    "config_file": "configs/train_configs/baseline_daicwoz_config.yaml",
    "model": "baseline",
    "use_modalities": ["daic_audio_mfcc", "daic_audio_egemaps", "daic_facial_aus", "daic_gaze", "daic_head_pose", "daic_text"],
    "n_temporal_windows": 1,
    "seconds_per_window": 6
})

# 加载配置和模型
cfg = load_config(args.config_file)
args = EasyDict({**cfg, **args})
from lib import nomenclature
from models.baseline_model import BaselineModel

# 初始化模型架构
model = BaselineModel(args)

# 加载权重
state_dict = torch.load(CHECKPOINT_PATH, map_location="cuda")
# 移除DataParallel包装的前缀
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval().cuda()

# 构造示例输入（匹配DAIC-WOZ模态维度）
dummy_inputs = {
    "daic_audio_mfcc": torch.randn(1, 100, 39).cuda(),    # (batch, seq_len, feature_dim)
    "daic_audio_egemaps": torch.randn(1, 100, 23).cuda(),
    "daic_facial_aus": torch.randn(1, 100, 20).cuda(),
    "daic_gaze": torch.randn(1, 100, 12).cuda(),
    "daic_head_pose": torch.randn(1, 100, 6).cuda(),
    "daic_text": torch.randn(1, 100, 768).cuda()
}

# 构造batch字典（匹配模型输入格式）
batch = {
    "video_frame_rate": torch.tensor([30.]).cuda(),
    "audio_frame_rate": torch.tensor([16000.]).cuda()
}
for mod_name, tensor in dummy_inputs.items():
    batch[f"modality:{mod_name}:data"] = tensor
    batch[f"modality:{mod_name}:mask"] = torch.ones_like(tensor[:, :, 0]).cuda()

# 导出ONNX
torch.onnx.export(
    model,
    (batch,),  # 模型输入
    ONNX_SAVE_PATH,
    input_names=["batch"],
    output_names=["depression_pred"],
    dynamic_axes={
        "modality:daic_audio_mfcc:data": {1: "seq_len"},
        "modality:daic_audio_egemaps:data": {1: "seq_len"},
        "modality:daic_facial_aus:data": {1: "seq_len"},
        "modality:daic_gaze:data": {1: "seq_len"},
        "modality:daic_head_pose:data": {1: "seq_len"},
        "modality:daic_text:data": {1: "seq_len"},
        "depression_pred": {0: "batch_size"}
    },
    opset_version=12,
    verbose=False
)

print(f"模型已导出至: {ONNX_SAVE_PATH}")

# 验证导出的模型
import onnxruntime as ort
ort_session = ort.InferenceSession(ONNX_SAVE_PATH)
# 转换输入为numpy格式
dummy_inputs_np = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
# 构造ONNX输入
ort_inputs = {f"modality:{k}:data": v for k, v in dummy_inputs_np.items()}
ort_inputs["video_frame_rate"] = np.array([30.], dtype=np.float32)
ort_inputs["audio_frame_rate"] = np.array([16000.], dtype=np.float32)
for k in dummy_inputs_np.keys():
    ort_inputs[f"modality:{k}:mask"] = np.ones_like(dummy_inputs_np[k][:, :, 0])

# 推理验证
ort_outputs = ort_session.run(None, ort_inputs)
print(f"ONNX模型推理输出形状: {ort_outputs[0].shape}")