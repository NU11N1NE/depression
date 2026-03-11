import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import random
import numpy as np

import lib.callbacks as callbacks
from lib.loggers import WandbLogger
from lib.arg_utils import define_args

from lib import NotALightningTrainer
from lib import nomenclature
from lib.forge import VersionCommand

from utils import get_cosine_schedule_with_warmup

VersionCommand().run()

args = define_args()

args.modalities = [modality for modality in args.modalities if modality.name in args.use_modalities]

# ========== 适配Kaggle：跳过wandb初始化 ==========
if args.mode != "offline":
    while True:
        try:
            wandb.init(project = 'perceiving-depression', group = args.group, entity = 'perceiving-depression')
            break
        except Exception as e:
            print(f"Wandb failed to initialize (Reason: {e}), retrying ... ")
    wandb.config.update(vars(args))
else:
    print("Offline mode: skip wandb initialization")
    wandb_logger = None  # 离线模式禁用wandb logger

# ========== 固定随机种子 ==========
if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# ========== 加载数据集和模型 ==========
dataset = nomenclature.DATASETS[args.dataset](args = args)
train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)

architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

# ========== 加载评估器 ==========
evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args)
    for evaluator_args in args.evaluators
]

# ========== 日志器适配 ==========
if args.mode == "offline":
    # 自定义简单日志器，替代wandb
    class SimpleLogger:
        def log(self, key, value):
            if key == "lr":
                print(f"Epoch {trainer.current_epoch} | LR: {value:.6f}")
            else:
                print(f"Epoch {trainer.current_epoch} | {key}: {value:.4f}")
    wandb_logger = SimpleLogger()
else:
    wandb_logger = WandbLogger()

# ========== 检查点配置 ==========
checkpoint_callback_best = callbacks.ModelCheckpoint(
    args = args,
    name = ' 🔥 Best Checkpoint Overall 🔥',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'/kaggle/working/checkpoints/{args.group}-{args.name}/best/',  # 保存到Kaggle工作目录
    save_best_only = True,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}',
)

checkpoint_callback_last = callbacks.ModelCheckpoint(
    args = args,
    name = '🛠️ Last Checkpoint 🛠️',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'/kaggle/working/checkpoints/{args.group}-{args.name}/last/',
    save_best_only = False,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}',
)

# ========== 优化器和调度器 ==========
if args.scheduler == "cosine":
    optimizer = model.configure_optimizers(lr = args.scheduler_args.max_lr)
    # 梯度累积适配
    scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_training_steps = args.epochs * len(train_dataloader) // args.accumulation_steps,
        num_warmup_steps = 0,
        last_epoch = -1
    )
else:
    raise NotImplementedError("Support only 'cosine' scheduler.")

# ========== 回调函数 ==========
lr_callback = callbacks.LambdaCallback(
    on_batch_end = lambda: scheduler.step() if (trainer.global_step + 1) % args.accumulation_steps == 0 else None
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: wandb_logger.log('lr', scheduler.get_last_lr()[0]) if (trainer.global_step + 1) % args.accumulation_steps == 0 else None
)

# 调试模式配置
if args.debug:
    print("[🐞DEBUG MODE🐞] Removing ModelCheckpoint ... ")
    checkpoint_callback_best.actually_save = False
    checkpoint_callback_last.actually_save = False
else:
    checkpoint_callback_best.actually_save = bool(args.save_model)
    checkpoint_callback_last.actually_save = bool(args.save_model)

callbacks = [
    checkpoint_callback_best,
    checkpoint_callback_last,
    lr_callback,
    lr_logger,
]

# ========== 训练器初始化 ==========
trainer = NotALightningTrainer(
    args = args,
    callbacks = callbacks,
    logger=wandb_logger,
)

# ========== 开始训练 ==========
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)