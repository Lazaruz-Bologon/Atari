# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import gymnasium as gym
import ale_py
import torch
from pathlib import Path
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ncps.torch import CfC
from ncps.datasets.torch import AtariCloningDataset
from atari_cell import *
class TrainingLogger:
    def __init__(self, log_dir, run_name):
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_file = os.path.join(log_dir, f"{run_name}_metrics.csv")
        self.batch_log_file = os.path.join(log_dir, f"{run_name}_batch_metrics.csv")
        self.metrics = []
        self.batch_metrics = []
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Initialize CSV headers
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy,mean_return,epoch_time,timestamp\n")
        if not os.path.exists(self.batch_log_file):
            with open(self.batch_log_file, 'w') as f:
                f.write("epoch,batch,batch_loss,batch_accuracy,global_step,timestamp\n")
    def log_epoch(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy, 
                 mean_return, epoch_time):
        """Log metrics for a training epoch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Save to metrics list
        metric = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'mean_return': mean_return,
            'epoch_time': epoch_time,
            'timestamp': timestamp
        }
        self.metrics.append(metric)
        # Append to CSV
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{train_accuracy},{val_loss},{val_accuracy},{mean_return},{epoch_time},{timestamp}\n")
    def log_batch(self, epoch, batch, batch_loss, batch_accuracy, global_step):
        """Log metrics for a training batch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Save to batch metrics list
        metric = {
            'epoch': epoch,
            'batch': batch,
            'batch_loss': batch_loss,
            'batch_accuracy': batch_accuracy,
            'global_step': global_step,
            'timestamp': timestamp
        }
        self.batch_metrics.append(metric)
        # Append to CSV
        with open(self.batch_log_file, 'a') as f:
            f.write(f"{epoch},{batch},{batch_loss},{batch_accuracy},{global_step},{timestamp}\n")
    def log_model_info(self, model_info):
        """Log model information to a JSON file"""
        info_file = os.path.join(self.log_dir, f"{self.run_name}_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=4)
    def create_visualizations(self, show=False):
        """Create visualizations from logged metrics"""
        # Load data
        df = pd.read_csv(self.log_file)
        batch_df = pd.read_csv(self.batch_log_file)
        # Set seaborn style
        sns.set_theme(style="whitegrid")
        # Create plots directory
        plots_dir = os.path.join(self.log_dir, f"{self.run_name}_plots")
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        # 1. Learning curves (loss and accuracy)
        plt.figure(figsize=(12, 10))
        # Training and validation loss
        plt.subplot(2, 1, 1)
        sns.lineplot(data=df, x='epoch', y='train_loss', marker='o', label='Training Loss', 
                    color='#1f77b4', linewidth=2)
        sns.lineplot(data=df, x='epoch', y='val_loss', marker='s', label='Validation Loss', 
                    color='#ff7f0e', linewidth=2)
        plt.title('Training and Validation Loss Over Time', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        # Training and validation accuracy
        plt.subplot(2, 1, 2)
        sns.lineplot(data=df, x='epoch', y='train_accuracy', marker='o', label='Training Accuracy', 
                    color='#2ca02c', linewidth=2)
        sns.lineplot(data=df, x='epoch', y='val_accuracy', marker='s', label='Validation Accuracy', 
                    color='#d62728', linewidth=2)
        plt.title('Training and Validation Accuracy Over Time', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'learning_curves.png'), dpi=300)
        # 2. Mean return over time
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='epoch', y='mean_return', marker='o', linewidth=2, 
                    color='#9467bd')
        plt.title('Mean Return Over Time', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Return', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'mean_return.png'), dpi=300)
        # 3. Batch loss during training
        plt.figure(figsize=(12, 6))
        g = sns.lineplot(data=batch_df, x='global_step', y='batch_loss', linewidth=1.5,
                       color='#8c564b', alpha=0.7)
        # Add smoothed line
        window_size = min(100, len(batch_df) // 10) if len(batch_df) > 0 else 1
        if len(batch_df) > window_size:
            batch_df['smooth_loss'] = batch_df['batch_loss'].rolling(window=window_size).mean()
            sns.lineplot(data=batch_df, x='global_step', y='smooth_loss', linewidth=2.5,
                       color='#e377c2', label=f'Moving Average (window={window_size})')
        plt.title('Batch Loss During Training', fontsize=16)
        plt.xlabel('Global Step', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'batch_loss.png'), dpi=300)
        # 4. Batch accuracy during training
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=batch_df, x='global_step', y='batch_accuracy', linewidth=1.5,
                   color='#7f7f7f', alpha=0.7)
        # Add smoothed line
        if len(batch_df) > window_size:
            batch_df['smooth_acc'] = batch_df['batch_accuracy'].rolling(window=window_size).mean()
            sns.lineplot(data=batch_df, x='global_step', y='smooth_acc', linewidth=2.5,
                       color='#bcbd22', label=f'Moving Average (window={window_size})')
        plt.title('Batch Accuracy During Training', fontsize=16)
        plt.xlabel('Global Step', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'batch_accuracy.png'), dpi=300)
        # 5. Training time per epoch
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='epoch', y='epoch_time', palette='viridis')
        plt.title('Training Time per Epoch', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'epoch_time.png'), dpi=300)
        if show:
            plt.show()
        else:
            plt.close('all')
        print(f"Visualizations saved to {plots_dir}")
        return plots_dir
def eval(model, valloader, criterion=nn.CrossEntropyLoss(), logger=None, epoch=None):
    losses, accs = [], []
    model.eval()
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(-1) == labels).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())    
    val_loss = np.mean(losses)
    val_acc = np.mean(accs)
    return val_loss, val_acc
def train_one_epoch(model, criterion, optimizer, trainloader, epoch, logger):
    running_loss = 0.0
    running_acc = 0.0
    pbar = tqdm(total=len(trainloader), ncols=80, desc="Training")
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        labels = labels.view(-1, *labels.shape[2:])  # flatten
        outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
        loss = criterion(outputs, labels)
        acc = (outputs.argmax(-1) == labels).float().mean().item()
        running_acc += acc        
        loss.backward()
        # Monitor gradients
        total_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.data.norm(2).item()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i + 1):0.4g}, acc={running_acc / (i + 1):0.4g}")
        pbar.update(1)
        # Log batch metrics
        global_step = epoch * len(trainloader) + i
        logger.log_batch(epoch, i, loss.item(), acc, global_step)
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = running_acc / len(trainloader)
    pbar.close()
    return epoch_loss, epoch_acc
def run_closed_loop(model, env, num_episodes=None):
    obs, info = env.reset()
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    with torch.no_grad():
        while True:
            # PyTorch require channel first images -> transpose data
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)
            # add batch and time dimension (with a single element in each)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            pred, hx = model(obs, hx)
            # remove time and batch dimension -> then argmax
            action = pred.squeeze(0).squeeze(0).argmax().item()
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += r
            if done:
                obs, info = env.reset()
                hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    # Count down the number of episodes
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns
def count_parameters(model):
    """Count trainable and total parameters in a model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
def analyze_dataset(dataset, name):
    """Analyze the distribution of actions in a dataset"""
    labels = []
    for _, label in tqdm(dataset, desc=f"Analyzing {name} Dataset"):
        labels.append(label.numpy())
    labels = np.concatenate(labels)
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {}
    print(f"{name} dataset label distribution:")
    for u, c in zip(unique, counts):
        percentage = 100 * c / len(labels)
        print(f"Action {u}: {c} samples ({percentage:.2f}%)")
        distribution[int(u)] = {
            "count": int(c),
            "percentage": float(percentage)
        }
    return distribution
def create_model(model_type, n_actions, hidden_size=64, connection_type="fc", sparsity_level=0.5, use_ode=False, dt=0.1):
    if connection_type.lower() not in ["fc", "ncp", "breakout", "modular", "small-world", "hierarchical", "distance-dependent"]:
        raise ValueError(f"Unsupported connection type: {connection_type}. Please use 'fc', 'ncp', 'small-world', 'breakout' or 'hierarchical'")    
    
    model_type = model_type.lower()
    connection_type = connection_type.lower()
    
    # 处理特殊的连接结构类型
    if connection_type == "breakout":
        if model_type == "ltc":
            return Conv3LTC_Breakout(n_actions, hidden_size)
        elif model_type == "wc":
            # 如果未来实现了WC的Breakout连接，可以在这里添加
            raise ValueError(f"Breakout connection structure currently not supported for WC models")
        else:
            raise ValueError(f"Breakout connection structure currently only supports LTC models")
    
    # 处理分层连接结构
    if connection_type == "modular" and model_type == "wc":
        return Conv3WilsonCowan_Modular(n_actions, hidden_size, sparsity_level, dt=dt, use_rk4=use_ode)
    if connection_type == "small-world" and model_type == "wc":
        return Conv3WilsonCowan_SmallWorld(n_actions, hidden_size, sparsity_level, dt=dt, use_rk4=use_ode)
    if connection_type == "hierarchical" and model_type == "wc":
        return Conv3WilsonCowan_Hierarchical(n_actions, hidden_size, sparsity_level, dt=dt, use_rk4=use_ode)
    if connection_type == "distance-dependent" and model_type == "wc":
        return Conv3WilsonCowan_DistanceDependent(n_actions, hidden_size, sparsity_level, dt=dt, use_rk4=use_ode)
    # Wilson-Cowan 模型处理
    if model_type == "wc":
        if connection_type == "fc":
            return Conv3WilsonCowan_FC(n_actions, hidden_size, dt=dt, use_rk4=use_ode)
        elif connection_type == "ncp":
            return Conv3WilsonCowan_Random(n_actions, hidden_size, sparsity_level, dt=dt, use_rk4=use_ode)
    
    # 其他模型的ODE变体
    if model_type in ["rnn", "lstm"] and use_ode:
        if model_type == "rnn":
            if connection_type == "fc":
                return Conv3RNN_FC(n_actions, hidden_size, use_ode=True)
            else:  # ncp
                return Conv3RNN_NCP(n_actions, hidden_size, sparsity_level, use_ode=True)
        elif model_type == "lstm":
            if connection_type == "fc":
                return Conv3LSTM_FC(n_actions, hidden_size, use_ode=True)
            else:  # ncp
                return Conv3LSTM_NCP(n_actions, hidden_size, sparsity_level, use_ode=True)
    
    # 标准模型处理
    if model_type == "cfc":
        if connection_type == "fc":
            return Conv3CfC_FC(n_actions, hidden_size)
        else:  # ncp
            return Conv3CfC_NCP(n_actions, hidden_size, sparsity_level)
    elif model_type == "ltc":
        if connection_type == "fc":
            return Conv3LTC_FC(n_actions, hidden_size)
        else:  # ncp
            return Conv3LTC_NCP(n_actions, hidden_size, sparsity_level)
    elif model_type == "lstm":
        if connection_type == "fc":
            return Conv3LSTM_FC(n_actions, hidden_size)
        else:  # ncp
            return Conv3LSTM_NCP(n_actions, hidden_size, sparsity_level)
    elif model_type == "rnn":
        if connection_type == "fc":
            return Conv3RNN_FC(n_actions, hidden_size)
        else:  # ncp
            return Conv3RNN_NCP(n_actions, hidden_size, sparsity_level)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please use 'cfc', 'ltc', 'lstm', 'rnn', or 'wc'")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Atari Game Model Training')
    parser.add_argument('--model', type=str, default='wc', choices=['cfc', 'ltc', 'lstm', 'rnn', 'wc'],
                        help='RNN type (default: cfc)')
    parser.add_argument('--connection', type=str, default='distance-dependent', choices=['fc', 'ncp', 'breakout', 'modular', 'small-world', 'hierarchical', 'distance-dependent'],
                        help='Connection type: fully connected (fc), neural circuit policies (ncp), breakout or modular (default: fc)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of game episodes for evaluation (default: 10)')
    parser.add_argument('--log_dir', type=str, default='./wclogs',
                        help='Log directory (default: ./logs)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Model save directory (default: ./checkpoints)')
    parser.add_argument('--use_ode', action='store_true',
                        help='Use ODE solving method (RK4 for Wilson-Cowan) (default: False)')
    parser.add_argument('--analyze_data', action='store_true',
                        help='Analyze dataset distribution before training (default: False)')
    parser.add_argument('--scheduler', action='store_true',
                        help='Use learning rate scheduler (default: False)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step for Wilson-Cowan integration (default: 0.1)')
    parser.add_argument('--excitatory_ratio', type=float, default=0.8,
                        help='Ratio of excitatory neurons in Wilson-Cowan model (default: 0.8)')
    parser.add_argument('--load_model', action='store_true',
                        help='Load pretrained model (default: False)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation without training (default: False)')
    parser.add_argument('--render', action='store_true',
                        help='Render game environment during evaluation (default: False)')
    parser.add_argument('--num_test_episodes', type=int, default=20,
                        help='Number of episodes to test loaded model (default: 20)')
    args = parser.parse_args()
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Set up run name and logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.connection}_{args.hidden_size}_{timestamp}"
    logger = TrainingLogger(args.log_dir, run_name)
    # Set up environment and datasets
    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)
    train_ds = AtariCloningDataset("breakout", split="train")
    val_ds = AtariCloningDataset("breakout", split="val")
    # Analyze dataset if requested
    if args.analyze_data:
        train_distribution = analyze_dataset(train_ds, "Training")
        val_distribution = analyze_dataset(val_ds, "Validation")
        # Save distribution data
        distribution_data = {
            "train_distribution": train_distribution,
            "val_distribution": val_distribution
        }
        with open(os.path.join(args.log_dir, f"{run_name}_data_distribution.json"), 'w') as f:
            json.dump(distribution_data, f, indent=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=4, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, shuffle=False)
    # Set up device    
    print(f"Using device: {device}")
    # Create model
    model = create_model(
        model_type=args.model,
        n_actions=env.action_space.n,
        hidden_size=args.hidden_size,
        connection_type=args.connection,
        use_ode=args.use_ode,
        sparsity_level=0.5,  # 可以通过参数控制
        dt=args.dt  # 添加 dt 参数
    )
    model=model.to(device)
    # Count parameters
    trainable_params, total_params = count_parameters(model)
    # Print model info
    ode_info = "Using ODE" if args.use_ode else "Not using ODE"
    print(f"Model: {args.model.upper()} Connection: {args.connection.upper()} Hidden Size: {args.hidden_size} {ode_info}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    # Log model info
    model_info = {
        "model_type": args.model.upper(),
        "connection_type": args.connection.upper(),
        "hidden_size": args.hidden_size,
        "use_ode": args.use_ode,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "timestamp": timestamp
    }
    logger.log_model_info(model_info)
    # Set up optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.load_model:
        if args.model_path is None:
            print("错误: --load_model 需要提供 --model_path 参数")
            sys.exit(1)
            
        print(f"加载预训练模型: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态（如果不是仅评估模式）
            if not args.eval_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"从 epoch {start_epoch} 开始继续训练")
                    
            # 打印加载的模型信息
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"加载的模型指标: 验证准确率={metrics['val_acc']*100:.2f}%, 平均回报={metrics['mean_return']:.2f}")
                
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)
    
    # 仅评估模式
    if args.eval_only:
        assert args.load_model, "仅评估模式需要加载预训练模型 (--load_model)"
        print("\n=== 仅评估模式 ===")
        
        # 创建环境（带渲染）
        test_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array" if args.render else None)
        test_env = wrap_deepmind(test_env)
        
        # 评估验证集性能
        val_loss, val_acc = eval(model, valloader, criterion)
        print(f"验证集准确率: {val_acc*100:.2f}%")
        
        # 运行游戏并评估
        print(f"运行 {args.num_test_episodes} 个测试回合...")
        returns = []
        episode_lengths = []
        
        for ep in range(args.num_test_episodes):
            obs, info = test_env.reset()
            hx = None  # 隐藏状态
            done = False
            episode_reward = 0
            steps = 0
            
            with torch.no_grad():
                while not done:
                    # 处理观察值
                    obs_tensor = np.transpose(obs, [2, 0, 1]).astype(np.float32)
                    obs_tensor = torch.from_numpy(obs_tensor).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 前向传播
                    pred, hx = model(obs_tensor, hx)
                    action = pred.squeeze(0).squeeze(0).argmax().item()
                    
                    # 执行动作
                    obs, r, term, trunc, _ = test_env.step(action)
                    done = term or trunc
                    episode_reward += r
                    steps += 1
                    
                    # 显示渲染画面（如果启用）
                    if args.render:
                        from IPython.display import clear_output, display
                        import PIL.Image
                        img = PIL.Image.fromarray(test_env.render())
                        clear_output(wait=True)
                        display(img)
                        time.sleep(0.02)  # 增加一点延迟使渲染可见
            
            returns.append(episode_reward)
            episode_lengths.append(steps)
            print(f"回合 {ep+1}/{args.num_test_episodes}: 分数={episode_reward}, 步数={steps}")
        
        # 打印测试结果统计
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(episode_lengths)
        
        print("\n=== 测试结果 ===")
        print(f"平均分数: {mean_return:.2f} ± {std_return:.2f}")
        print(f"平均回合长度: {mean_length:.1f} 步")
        print(f"最高分: {np.max(returns)}")
        print(f"最低分: {np.min(returns)}")
        
        # 可视化分数分布
        if len(returns) >= 5:  # 只有当有足够样本时才绘图
            plt.figure(figsize=(10, 6))
            sns.histplot(returns, kde=True)
            plt.title(f'测试分数分布 ({args.num_test_episodes} 回合)', fontsize=16)
            plt.xlabel('分数', fontsize=14)
            plt.ylabel('频率', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            hist_path = os.path.join(args.log_dir, f"score_histogram_{timestamp}.png")
            plt.savefig(hist_path, dpi=300)
            print(f"分数分布直方图已保存至: {hist_path}")
        
        sys.exit(0)
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    # Training loop
    start_time = time.time()
    best_val_acc = 0.0
    best_model_path = None
    for epoch in range(start_epoch,args.epochs):
        # Train
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, trainloader, epoch, logger)
        # Evaluate
        val_loss, val_acc = eval(model, valloader, criterion, logger, epoch)
        # Run game environment
        returns = run_closed_loop(model, env, num_episodes=args.eval_episodes)
        mean_return = np.mean(returns) if returns else 0.0
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        # Log epoch metrics
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, mean_return, epoch_time)
        # Update learning rate if scheduler is enabled
        if args.scheduler:
            scheduler.step(val_loss)
        # Print status
        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, train_acc={train_acc*100:0.2f}%, "
              f"val_loss={val_loss:0.4g}, val_acc={val_acc*100:0.2f}%, "
              f"mean_return={mean_return:.2f}, time={epoch_time:.2f}s")
        # Save best model
        curr_model_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'mean_return': mean_return,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(
                args.checkpoint_dir,
                f"best_model_{args.model}_{args.connection}_{args.hidden_size}_{timestamp}.pt"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': curr_model_metrics
            }, best_model_path)
            print(f"Best model saved! Validation accuracy: {best_val_acc*100:.2f}%")
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Print final info
    print(f"\nTraining complete! Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best model saved to {best_model_path}")
    # Add training time to model info
    model_info["training_time"] = {
        "total_seconds": total_time,
        "formatted": f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    }
    logger.log_model_info(model_info)
    # Create visualizations
    plots_dir = logger.create_visualizations()
    print(f"Log files saved to {args.log_dir}")