"""
NanoGPT Configuration

Weight Tying: Disabled
Query Mode: original
  - Q = X @ W_Q (original query weights)
Model Size: base (n_embd=768, mlp_factor=4)

Checkpoint saves at steps: [50000, 100000, 105000]
Training stops at step: 105001
"""

import math

model_args = {
    # Model architecture
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "num_heads": 12,
    "n_embd": 768,
    "head_size": 64,
    "mlp_hidden_size": 768 * 4,

    # Weight configuration
    "tie_weights": False,

    # Query mode: "original", "identity", "residual", "residual_gelu"
    "query_mode": "original",

    # Regularization
    "dropout": 0.0,
    "bias": False,

    # Training batch configuration
    "batch_size": 12,
    "accumulation_size": 40,

    # Attention scale
    "scale": 1/(math.sqrt(768//12)),

    # Optimizer settings
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Learning rate schedule
    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,

    # Checkpoint and early stopping
    "save_checkpoint_steps": [50000, 100000, 105000],
    "max_iters": 105001,
}
