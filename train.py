import sys
import os

# Parse GPU argument early (before torch import)
if '--gpu' in sys.argv:
    gpu_idx = sys.argv[sys.argv.index('--gpu') + 1]
    sys.argv.remove('--gpu')
    sys.argv.remove(gpu_idx)
    if gpu_idx != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd
from model import GPTConfig, GPT
from logger import get_logger

import importlib.util

# Get config from command line: python train.py configs_untied/config_untiedw_identity
config_arg = sys.argv[1] if len(sys.argv) > 1 else "config_original_untiedw"
config_path = config_arg if config_arg.endswith('.py') else f"{config_arg}.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
model_args = config_module.model_args

# Derive output directory from config name (strip path and .py)
config_name = os.path.basename(config_path).replace('.py', '')
out_dir = f'out_{config_name}'

eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' + '_run_' + str(time.time())
# data
dataset = 'openwebtext'
batch_size = 12
gradient_accumulation_steps = 5 * 8 * 12//batch_size # used to simulate larger batch sizes
 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024


learning_rate = model_args.get("learning_rate", 6e-4) # max learning rate
max_iters = model_args.get("max_iters", 600000)
 # total number of training iterations
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = model_args.get("lr_decay_iters", 600000) # should be ~= max_iters per Chinchilla
min_lr = model_args.get("min_lr", learning_rate/10) # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# logging
if wandb_log and master_process:
    import wandb
    # Using offline mode for wandb to log locally; this avoids network calls and allows syncing logs later.
    # This is useful when running on clusters without internet access or when you want to control when logs are uploaded.
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode = "offline")
    wandb_dir = wandb.run.dir 
    wandb.log({"model_args": model_args})

logger = get_logger()
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'model_args.pkl'), 'wb') as f:
        pickle.dump(model_args, f)
logger.info("Starting training...")

#we will collect the training and eval info in a gpu torch tensor in order to avoid syncing between gpu-cpu


# Initialize collectors (default for scratch)
train_collector = torch.zeros((max_iters+1, 3), device="cuda:0", dtype=torch.float32)
eval_train_collector = torch.zeros((max_iters//eval_interval + 1, 3), device="cuda:0", dtype=torch.float32)
eval_val_collector = torch.zeros((max_iters//eval_interval + 1, 3), device="cuda:0", dtype=torch.float32)

# Load existing tensors if resuming
tensor_path = os.path.join(out_dir, 'tensor.pt')
eval_train_path = os.path.join(out_dir, 'eval_train_tensor.pt')
eval_val_path = os.path.join(out_dir, 'eval_val_tensor.pt')

if os.path.exists(tensor_path):
    train_collector = torch.load(tensor_path, map_location="cuda:0")
if os.path.exists(eval_train_path):
    eval_train_collector = torch.load(eval_train_path, map_location="cuda:0")
if os.path.exists(eval_val_path):
    eval_val_collector = torch.load(eval_val_path, map_location="cuda:0")

# various inits, derived attributes, I/O setup

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

train_indices = np.load(os.path.join(data_dir, 'train_indices.npy'))#.reshape(max_iters, gradient_accumulation_steps, model_args['batch_size'])
val_indices = np.load(os.path.join(data_dir, 'val_indices.npy'))#.reshape(max_iters, gradient_accumulation_steps, model_args['batch_size'])
eval_train_indices = np.load(os.path.join(data_dir, 'eval_train_indices.npy'))#.reshape(max_iters, gradient_accumulation_steps, model_args['batch_size'])
eval_val_indices = np.load(os.path.join(data_dir, 'eval_val_indices.npy'))#.reshape(max_iters, gradient_accumulation_steps, model_args['batch_size'])
def get_batch(split, it, microstep):
    # Use pre-defined indices for train split
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        # Select indices for current iteration and microstep
        ix = train_indices[it, microstep*batch_size:(microstep+1)*batch_size].flatten()
        # ix is shape (batch_size,)
    elif split =="val":
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # Select indices for current iteration and microstep
        ix = val_indices[it, microstep*batch_size:(microstep+1)*batch_size].flatten()
        # ix is shape (batch_size,)
    elif split =="eval_train":
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        # Select indices for current iteration and microstep
        ix = eval_train_indices[it, microstep*batch_size:(microstep+1)*batch_size].flatten()
        # ix is shape (batch_size,)
    elif split =="eval_val":
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # Select indices for current iteration and microstep
        ix = eval_val_indices[it, microstep*batch_size:(microstep+1)*batch_size].flatten()
        # ix is shape (batch_size,)
    # Prepare batches
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    logger.info(gptconf.__dict__)
    model = GPT(gptconf)
    logger.info(f"Number of parameters in the model: {model.get_num_params(non_embedding=False):,}")
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'num_heads', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    logger.info(gptconf.__dict__)
    logger.info(f"Number of parameters in the model: {model.get_num_params(non_embedding=False):,}")
    

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=model_args["dropout"])
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'num_heads', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))

# optimizer
optimizer = model.configure_optimizers(model_args["weight_decay"], learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(iter_num=0):
    out = {}
    model.eval()
    for split in ['eval_train', 'eval_val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, it = iter_num, microstep = k)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        if split == 'eval_train':
            eval_train_collector[iter_num,0] = eval_iters
            eval_train_collector[iter_num,1] = losses.mean()
            eval_train_collector[iter_num,2] = losses.std()
        else:
            eval_val_collector[iter_num,0] = eval_iters
            eval_val_collector[iter_num,1] = losses.mean()
            eval_val_collector[iter_num,2] = losses.std()


    torch.save(eval_train_collector, os.path.join(out_dir, "eval_train_tensor.pt"))
    torch.save(eval_val_collector, os.path.join(out_dir, "eval_val_tensor.pt"))
    torch.save(train_collector, os.path.join(out_dir, "tensor.pt"))

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch('train', it=iter_num, microstep=0)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
save_checkpoint_steps = model_args.get("save_checkpoint_steps", [])

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(iter_num=iter_num//eval_interval)
            print(f"step {iter_num}: train loss {losses['eval_train']:.4f}, val loss {losses['eval_val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['eval_train'],
                    "val/loss": losses['eval_val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                })
            if losses['eval_val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['eval_val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            
            # Save named checkpoint at specific steps (always, regardless of best_val_loss)
            if iter_num in save_checkpoint_steps:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if micro_step != gradient_accumulation_steps - 1:
            X, Y = get_batch('train',it = iter_num, microstep = micro_step+1)
        else:
            X, Y = get_batch('train',it = iter_num+1, microstep = 0)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        train_collector[iter_num,0] = lossf
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()