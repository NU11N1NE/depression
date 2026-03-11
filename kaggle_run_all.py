import os
import subprocess

# 1. 安装依赖
print("安装依赖...")
subprocess.run([
    "pip", "install", "torch==2.3.1", "pandas", "tqdm", "transformers",
    "einops", "easydict", "pyyaml", "onnx", "onnxruntime-gpu", "scipy"
])

# 2. 特征提取
print("提取文本特征...")
subprocess.run(["python", "scripts/feature_extraction/daicwoz/prepare_text.py",
                "--src-root", "/kaggle/input/daic-woz-raw/no-chunked/text",
                "--dest-root", "/kaggle/working/daic-woz/data"])

print("提取共振峰特征...")
subprocess.run(["python", "scripts/feature_extraction/daicwoz/prepare_formant.py",
                "--src-root", "/kaggle/input/daic-woz-raw/data",
                "--dest-root", "/kaggle/working/daic-woz/no-chunked"])

# 3. 训练模型
print("开始训练...")
subprocess.run(["python", "train_kaggle.py"])

# 4. 评估模型
print("开始评估...")
subprocess.run(["python", "evaluate_kaggle.py"])

# 5. 导出模型（需手动替换检查点路径）
print("导出模型...")
subprocess.run(["python", "export_model.py"])

print("所有步骤执行完成！")