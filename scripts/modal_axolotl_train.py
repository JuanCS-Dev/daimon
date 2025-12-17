"""
NOESIS Fine-Tuning com Axolotl + Modal Labs
============================================

Baseado 100% no exemplo oficial: https://github.com/modal-labs/llm-finetuning

Usa:
- Axolotl para multi-GPU training (DeepSpeed ZeRO-2)
- 8x H100 GPUs
- Dataset 300k exemplos
"""

import os
from datetime import datetime
from pathlib import Path, PurePosixPath
import secrets
from typing import Union

import modal

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

APP_NAME = "noesis-axolotl-train"
MINUTES = 60
HOURS = 60 * MINUTES

# Imagem oficial Axolotl 2025 - PyTorch 2.6.0, Python 3.11, CUDA 12.4
# Fonte: https://docs.axolotl.ai/docs/docker.html
# NÃO fazer pip_install - a imagem já tem todas as dependências corretas
axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:main-latest")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false",
            AXOLOTL_NCCL_TIMEOUT="120",
        )
    )
    .entrypoint([])
)

# Volumes
pretrained_volume = modal.Volume.from_name("noesis-pretrained-vol", create_if_missing=True)
runs_volume = modal.Volume.from_name("noesis-runs-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("noesis-training-data", create_if_missing=True)

VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
    "/data": data_volume,
}

# App com secrets
app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
    ],
)

# GPU Config - 8x H100
GPU_CONFIG = "H100:8"

# ============================================================================
# CONFIG YAML PARA LLAMA 3.1 8B
# ============================================================================

AXOLOTL_CONFIG = """
# =============================================================================
# NOESIS Philosophical Training - Llama 3.1 8B
# =============================================================================

base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

# Sequence length
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

# LoRA Configuration
adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out: false

# Dataset
datasets:
  - path: /data/train_300k.jsonl
    ds_type: json
    type:
      field_instruction: prompt
      field_output: response
      format: |
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {output}<|eot_id|>

# Validation
val_set_size: 0.01

# Training
output_dir: ./lora-out
num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.03
optimizer: adamw_torch

# DeepSpeed for multi-GPU
deepspeed: /root/deepspeed_zero2.json

# Performance
bf16: true
tf32: true
gradient_checkpointing: true
flash_attention: true

# Logging
logging_steps: 10
save_strategy: steps
save_steps: 500
eval_steps: 500

# Special tokens for Llama 3.1
special_tokens:
  pad_token: <|end_of_text|>

# Misc
seed: 42
strict: false
"""

DEEPSPEED_CONFIG = """{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
"""

# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def run_cmd(cmd: str, run_folder: str):
    """Executa comando com reload/commit de volumes."""
    import subprocess

    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()
    VOLUME_CONFIG["/data"].reload()

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=run_folder)

    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        raise RuntimeError(f"Command failed: {cmd}")

    VOLUME_CONFIG["/runs"].commit()


@app.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def train(run_folder: str):
    """Executa treinamento com 8 GPUs via DeepSpeed."""
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"

    print("=" * 70)
    print("  NOESIS PHILOSOPHICAL TRAINING - AXOLOTL + DEEPSPEED")
    print("=" * 70)
    print(f"  GPUs: {gpu_count}x {gpu_name}")
    print(f"  Run folder: {run_folder}")
    print("=" * 70)

    # Comando Axolotl com accelerate para multi-GPU
    cmd = f"accelerate launch --num_processes {gpu_count} --num_machines 1 --mixed_precision bf16 --dynamo_backend no -m axolotl.cli.train ./config.yml"

    run_cmd(cmd, run_folder)

    print("Training complete!")
    return {"status": "success", "gpus": gpu_count}


@app.function(
    image=axolotl_image,
    gpu="A10G:1",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def merge_lora(run_folder: str):
    """Merge LoRA weights into base model."""
    import torch

    print("Merging LoRA weights...")

    cmd = f"accelerate launch --num_processes 1 -m axolotl.cli.merge_lora ./config.yml --lora_model_dir='./lora-out'"
    run_cmd(cmd, run_folder)

    VOLUME_CONFIG["/runs"].commit()
    print("Merge complete!")


@app.function(
    image=axolotl_image,
    volumes=VOLUME_CONFIG,
    timeout=30 * MINUTES,
)
def setup_training():
    """Prepara ambiente de treinamento."""
    from huggingface_hub import snapshot_download
    import os

    print("=" * 70)
    print("  SETUP: Downloading model and preparing configs")
    print("=" * 70)

    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in environment!")

    # Download base model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        snapshot_download(model_name, local_files_only=True, token=hf_token)
        print(f"Model {model_name} already cached.")
    except:
        print(f"Downloading {model_name}...")
        snapshot_download(model_name, token=hf_token)
        VOLUME_CONFIG["/pretrained"].commit()
        print("Model downloaded and cached.")

    # Create run folder
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"noesis-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    # Write config
    with open(f"{run_folder}/config.yml", "w") as f:
        f.write(AXOLOTL_CONFIG)

    # Write DeepSpeed config
    with open("/root/deepspeed_zero2.json", "w") as f:
        f.write(DEEPSPEED_CONFIG)

    VOLUME_CONFIG["/runs"].commit()

    print(f"Run folder created: {run_folder}")
    return run_folder


@app.local_entrypoint()
def main():
    """Entry point principal."""
    print("=" * 70)
    print("  NOESIS AXOLOTL TRAINING - 8x H100 GPUs")
    print("=" * 70)

    # 1. Setup
    print("\n[1/3] Setting up training environment...")
    run_folder = setup_training.remote()
    print(f"Run folder: {run_folder}")

    # 2. Train
    print("\n[2/3] Starting training on 8x H100...")
    result = train.remote(run_folder)
    print(f"Training result: {result}")

    # 3. Merge
    print("\n[3/3] Merging LoRA weights...")
    merge_lora.remote(run_folder)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print(f"  Output: {run_folder}")
    print("=" * 70)
