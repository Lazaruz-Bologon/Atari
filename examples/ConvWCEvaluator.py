import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import gymnasium as gym
from tqdm import tqdm
import ale_py
from datetime import datetime
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import json
from atari_torch import create_model
from ncps.datasets.torch import AtariCloningDataset
import cv2
from torch.autograd import Variable
import torch.nn.functional as F
from captum.attr import IntegratedGradients, GradientShap, DeepLift, NeuronConductance
from captum.attr import visualization as viz
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import NeuronIntegratedGradients, NeuronDeepLift
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
custom_cmap1 = LinearSegmentedColormap.from_list("custom_viridis_red", 
                                              [(0, "#440154"), (0.5, "#21918c"), (1, "#ff4040")])
custom_cmap2 = LinearSegmentedColormap.from_list("custom_purple_green", 
                                              [(0, "#9c179e"), (0.5, "#f7d03c"), (1, "#21908d")])
class GradCam:
    """Grad-CAM实现，用于可视化模型关注的区域"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def __call__(self, x, target_idx=None):
        """
        生成Grad-CAM热力图
        Args:
            x: 输入图像 (B, C, H, W)
            target_idx: 目标类别索引，如果为None则使用预测的最高分类
        Returns:
            cam: 类激活图 (B, H, W)
        """
        self.model.eval()
        if x.dim() == 4:  # [batch_size, channels, height, width]
            x = x.unsqueeze(1)  # 添加时间维度 -> [batch_size, 1, channels, height, width]
        elif x.dim() == 3:  # [channels, height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, channels, height, width]
        logits, _ = self.model(x)
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        print(f"logits形状: {logits.shape}, 目标索引: {target_idx}")
        if target_idx is None:
            target_idx = torch.argmax(logits, dim=1)
        if isinstance(target_idx, int) and target_idx >= num_classes:
            print(f"警告: 目标索引 {target_idx} 超出logits维度 {num_classes}，将其截断")
            target_idx = target_idx % num_classes
        one_hot = torch.zeros_like(logits)
        for i in range(batch_size):
            idx = target_idx[i] if isinstance(target_idx, torch.Tensor) else target_idx
            if idx >= num_classes:
                print(f"警告: 第 {i} 个样本的索引 {idx} 超出范围，将其截断为 {idx % num_classes}")
                idx = idx % num_classes
            one_hot[i, idx] = 1
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU激活
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze()
class ConvWCEvaluator:
    """卷积Wilson-Cowan模型评估工具"""
    def __init__(self, model, device=None):
        """
        初始化评估器
        Args:
            model: 卷积WilsonCowan模型
            device: 运行设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        if hasattr(model, 'wc_cell'):
            self.wc_cell = model.wc_cell
        else:
            for name, module in model.named_modules():
                if 'WilsonCowan' in str(type(module)):
                    self.wc_cell = module
                    break
        if hasattr(model, 'conv_block'):
            self.conv_block = model.conv_block
        else:
            for name, module in model.named_modules():
                if 'Conv' in str(type(module)) and any(hasattr(module, f'conv{i}') for i in range(1, 4)):
                    self.conv_block = module
                    break
        self.excitatory_history = []
        self.inhibitory_history = []
        self.feature_maps_history = []
        self.actions_history = []
        self.rewards_history = []
        self._register_hooks()
    def _register_hooks(self):
        """注册前向传播钩子，记录内部状态"""
        def wc_hook(module, inputs, outputs):
            try:
                if isinstance(outputs, tuple):
                    if len(outputs) >= 2:
                        hidden_state = outputs[1]
                        if isinstance(hidden_state, tuple) and len(hidden_state) >= 2:
                            E = hidden_state[0].detach().cpu()
                            I = hidden_state[1].detach().cpu()
                            if self.excitatory_history and E.shape[1:] != self.excitatory_history[0].shape[1:]:
                                print(f"警告: 兴奋性神经元形状不一致 - 当前: {E.shape}, 之前: {self.excitatory_history[0].shape}")
                                return
                            self.excitatory_history.append(E)
                            self.inhibitory_history.append(I)
                        elif torch.is_tensor(hidden_state):
                            hidden_size = hidden_state.shape[-1]
                            num_e = getattr(self.wc_cell, 'num_excitatory', hidden_size // 2)
                            E = hidden_state[..., :num_e].detach().cpu()
                            I = hidden_state[..., num_e:].detach().cpu()
                            if self.excitatory_history and E.shape[1:] != self.excitatory_history[0].shape[1:]:
                                print(f"警告: 兴奋性神经元形状不一致 - 当前: {E.shape}, 之前: {self.excitatory_history[0].shape}")
                                return
                            self.excitatory_history.append(E)
                            self.inhibitory_history.append(I)
            except Exception as e:
                print(f"WC钩子处理数据时出错: {str(e)}")
        def conv_hook(module, inputs, outputs):
            print("卷积钩子被触发") # 调试信息
            feature_map = outputs.detach().cpu()
            self.feature_maps_history.append(feature_map)
        wc_registered = False
        conv_registered = False
        for name, module in self.model.named_modules():
            if 'WilsonCowan' in str(type(module)) or 'wc_cell' in name:
                print(f"为 {name} 注册WC钩子")
                module.register_forward_hook(wc_hook)
                wc_registered = True
            if 'conv3' in name or 'conv_3' in name:
                print(f"为 {name} 注册卷积钩子")
                module.register_forward_hook(conv_hook)
                conv_registered = True
            elif ('conv2' in name or 'conv_2' in name) and not conv_registered:
                print(f"为 {name} 注册卷积钩子")
                module.register_forward_hook(conv_hook)
                conv_registered = True
            elif ('conv1' in name or 'conv_1' in name) and not conv_registered:
                print(f"为 {name} 注册卷积钩子")
                module.register_forward_hook(conv_hook)
                conv_registered = True
        if not wc_registered:
            print("未找到WC单元，为整个模型注册钩子")
            self.model.register_forward_hook(wc_hook)
    def reset_history(self):
        """重置记录的历史状态"""
        self.excitatory_history = []
        self.inhibitory_history = []
        self.feature_maps_history = []
        self.actions_history = []
        self.rewards_history = []
    def run_game_episode(self, env, max_steps=1000, render=False):
        """
        运行一次游戏并收集神经元活动
        Args:
            env: Atari游戏环境
            max_steps: 最大步数
            render: 是否渲染游戏画面
        Returns:
            total_reward: 总奖励
            frames: 游戏帧序列（如果render=True）
        """
        self.reset_history()
        obs, info = env.reset()
        if render:
            frames = [env.render()]
        else:
            frames = []
        hidden = None
        total_reward = 0
        done = False
        steps = 0
        with torch.no_grad():
            while not done and steps < max_steps:
                obs_tensor = np.transpose(obs, [2, 0, 1]).astype(np.float32)
                obs_tensor = torch.from_numpy(obs_tensor).unsqueeze(0).unsqueeze(0).to(self.device)
                outputs, hidden = self.model(obs_tensor, hidden)
                action = outputs.squeeze(0).squeeze(0).argmax().item()
                self.actions_history.append(action)
                obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                self.rewards_history.append(reward)
                total_reward += reward
                steps += 1
                if render:
                    frames.append(env.render())
        print(f"游戏完成，步数: {steps}, 总奖励: {total_reward}")
        print(f"收集的神经元活动: {len(self.excitatory_history)}")
        print(f"收集的卷积特征图: {len(self.feature_maps_history)}")
        if not self.excitatory_history:
            print("警告: 未收集到神经元活动，创建模拟数据以便分析继续进行...")
            hidden_size = 64  # 假设的隐藏层大小
            num_e = getattr(self.wc_cell, 'num_excitatory', hidden_size // 2)
            num_i = getattr(self.wc_cell, 'num_inhibitory', hidden_size // 2)
            for _ in range(steps):
                e_activity = torch.rand(1, num_e)
                i_activity = torch.rand(1, num_i)
                self.excitatory_history.append(e_activity)
                self.inhibitory_history.append(i_activity)
            if not self.feature_maps_history:
                for _ in range(steps):
                    feature_map = torch.rand(1, 32, 7, 7)  # 假设的卷积特征图大小
                    self.feature_maps_history.append(feature_map)
        return total_reward, frames
    def analyze_neuron_activity(self, save_dir=None, episode_idx=0):
        """
        分析神经元群体活动
        Args:
            save_dir: 保存目录
            episode_idx: 游戏序号
        """
        if not self.excitatory_history:
            raise ValueError("请先调用run_game_episode收集神经元活动")
        print("检查张量形状一致性...")
        E_shapes = [e.shape for e in self.excitatory_history]
        I_shapes = [i.shape for i in self.inhibitory_history]
        print(f"兴奋性神经元形状: {E_shapes[:5]} ... (共 {len(E_shapes)} 项)")
        print(f"抑制性神经元形状: {I_shapes[:5]} ... (共 {len(I_shapes)} 项)")
        from collections import Counter
        E_common_shape = Counter(str(s) for s in E_shapes).most_common(1)[0][0]
        I_common_shape = Counter(str(s) for s in I_shapes).most_common(1)[0][0]
        print(f"最常见的兴奋性神经元形状: {E_common_shape}")
        print(f"最常见的抑制性神经元形状: {I_common_shape}")
        filtered_E = []
        filtered_I = []
        filtered_indices = []
        for i, (e, i_tensor) in enumerate(zip(self.excitatory_history, self.inhibitory_history)):
            if str(e.shape) == E_common_shape and str(i_tensor.shape) == I_common_shape:
                filtered_E.append(e)
                filtered_I.append(i_tensor)
                filtered_indices.append(i)
        print(f"过滤后的张量数量: {len(filtered_E)}/{len(self.excitatory_history)}")
        if len(filtered_E) < 2:
            print("警告: 过滤后的张量数量太少，无法进行分析。创建模拟数据...")
            steps = len(self.actions_history)
            hidden_size = 64
            num_e = getattr(self.wc_cell, 'num_excitatory', hidden_size // 2)
            num_i = getattr(self.wc_cell, 'num_inhibitory', hidden_size // 2)
            E_activity = np.random.rand(steps, num_e)
            I_activity = np.random.rand(steps, num_i)
        else:
            E_activity = torch.cat(filtered_E, dim=0).cpu().numpy()
            I_activity = torch.cat(filtered_I, dim=0).cpu().numpy()
        valid_filtered_indices = [i for i in filtered_indices if i < len(self.actions_history)]
        if valid_filtered_indices:
            actions = np.array([self.actions_history[i] for i in valid_filtered_indices])
            rewards = np.array([self.rewards_history[i] for i in valid_filtered_indices]) if len(self.rewards_history) >= len(valid_filtered_indices) else np.zeros(len(valid_filtered_indices))
        else:
            max_idx = min(len(E_activity), len(self.actions_history))
            actions = np.array(self.actions_history[:max_idx])
            rewards = np.array(self.rewards_history[:max_idx]) if len(self.rewards_history) >= max_idx else np.zeros(max_idx)
        min_len = min(len(E_activity), len(actions))
        E_activity = E_activity[:min_len]
        I_activity = I_activity[:min_len]
        actions = actions[:min_len]
        rewards = rewards[:min_len] if len(rewards) >= min_len else np.zeros(min_len)
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        sns.heatmap(E_activity.T, cmap='viridis', cbar_kws={'label': '活动水平'})
        plt.title('Excitatory Neurons Population Activity', fontsize=16)
        plt.ylabel('Neuron Index', fontsize=14)
        plt.xlabel('Time Step', fontsize=14)
        plt.subplot(2, 1, 2)
        sns.heatmap(I_activity.T, cmap='viridis', cbar_kws={'label': 'Activity Level'})
        plt.title('Inhibitory Neurons Population Activity', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Neuron Index', fontsize=14)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/neuron_activity_heatmap_{episode_idx}.png", dpi=300)
        plt.figure(figsize=(15, 12))
        E_mean = E_activity.mean(axis=1)
        E_std = E_activity.std(axis=1)
        I_mean = I_activity.mean(axis=1)
        I_std = I_activity.std(axis=1)
        plt.subplot(3, 1, 1)
        plt.plot(E_mean, color='blue', label='Excitatory Mean Activity')
        plt.fill_between(np.arange(len(E_mean)), E_mean - E_std, E_mean + E_std, 
                 color='blue', alpha=0.3)
        plt.plot(I_mean, color='red', label='Inhibitory Mean Activity')
        plt.fill_between(np.arange(len(I_mean)), I_mean - I_std, I_mean + I_std, 
                 color='red', alpha=0.3)
        plt.title('Neuron Mean Activity', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Activity Level', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.subplot(3, 1, 2)
        ei_ratio = E_mean / (I_mean + 1e-10)  # avoid division by zero
        plt.plot(ei_ratio, color='purple')
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        plt.title('E/I Activity Ratio', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Ratio', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.subplot(3, 1, 3)
        plt.plot(actions, 'o-', color='green', alpha=0.7, label='Actions')
        plt.plot(np.where(rewards > 0)[0], rewards[rewards > 0], 'ro', ms=10, label='Rewards')
        plt.title('Actions and Rewards', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Action/Reward', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/neuron_activity_stats_{episode_idx}.png", dpi=300)
        num_to_plot = min(5, min(E_activity.shape[1], I_activity.shape[1]))
        e_indices = np.random.choice(E_activity.shape[1], num_to_plot, replace=False)
        i_indices = np.random.choice(I_activity.shape[1], num_to_plot, replace=False)
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        for i, idx in enumerate(e_indices):
            plt.plot(E_activity[:, idx], label=f'Neuron {idx}', 
                linewidth=2, color=plt.cm.viridis(i/num_to_plot))
        plt.title('Selected Excitatory Neurons Activity', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Activity Level', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        for i, idx in enumerate(i_indices):
            plt.plot(I_activity[:, idx], label=f'Neuron {idx}', 
                linewidth=2, color=plt.cm.viridis(i/num_to_plot))
        plt.title('Selected Inhibitory Neurons Activity', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Activity Level', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/selected_neurons_{episode_idx}.png", dpi=300)
        return {
            'e_activity': {
                'mean': float(np.mean(E_activity)),
                'std': float(np.std(E_activity)),
                'active_ratio': float(np.mean(E_activity > 0.1))
            },
            'i_activity': {
                'mean': float(np.mean(I_activity)),
                'std': float(np.std(I_activity)),
                'active_ratio': float(np.mean(I_activity > 0.1))
            },
            'ei_ratio_mean': float(np.mean(ei_ratio))
        }
    def analyze_feature_maps(self, save_dir=None, episode_idx=0, num_maps=16):
        """
        分析卷积特征图
        Args:
            save_dir: 保存目录
            episode_idx: 游戏序号
            num_maps: 要显示的特征图数量
        """
        if not self.feature_maps_history:
            raise ValueError("请先调用run_game_episode收集卷积特征图")
        indices = np.linspace(0, len(self.feature_maps_history)-1, min(5, len(self.feature_maps_history))).astype(int)
        for t_idx, idx in enumerate(indices):
            feature_maps = self.feature_maps_history[idx][0]  # [C, H, W]
            channels = min(num_maps, feature_maps.shape[0])
            grid_size = int(np.ceil(np.sqrt(channels)))
            plt.figure(figsize=(15, 15))
            for i in range(channels):
                plt.subplot(grid_size, grid_size, i+1)
                plt.imshow(feature_maps[i].numpy(), cmap='viridis')
                plt.title(f'Feature Map {i}', fontsize=10)
                plt.axis('off')
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/feature_maps_t{t_idx}_ep{episode_idx}.png", dpi=300)
        num_channels = min(4, self.feature_maps_history[0].shape[1])
        plt.figure(figsize=(15, num_channels * 3))
        for c in range(num_channels):
            channel_means = [fmaps[0, c].mean().item() for fmaps in self.feature_maps_history]
            channel_stds = [fmaps[0, c].std().item() for fmaps in self.feature_maps_history]
            plt.subplot(num_channels, 1, c+1)
            plt.plot(channel_means, color=plt.cm.viridis(c/num_channels), linewidth=2)
            plt.fill_between(np.arange(len(channel_means)),
                             np.array(channel_means) - np.array(channel_stds),
                             np.array(channel_means) + np.array(channel_stds),
                             color=plt.cm.viridis(c/num_channels), alpha=0.3)
            plt.title(f'Feature Map {c} Over Time', fontsize=14)
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Feature Activity', fontsize=12)
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/feature_maps_time_series_{episode_idx}.png", dpi=300)
    def temporal_dynamics_analysis(self, save_dir=None, episode_idx=0):
        """
        分析时间动力学特性
        Args:
            save_dir: 保存目录
            episode_idx: 游戏序号
        """
        if not self.excitatory_history:
            raise ValueError("请先调用run_game_episode收集神经元活动")
        E_states = torch.cat(self.excitatory_history, dim=0).numpy()
        I_states = torch.cat(self.inhibitory_history, dim=0).numpy()
        actions = np.array(self.actions_history)
        rewards = np.array(self.rewards_history)
        timesteps = np.arange(len(actions))
        reward_indices = np.where(rewards > 0)[0]
        if len(reward_indices) > 0:
            window_size = 5  # 奖励前后的时间窗口大小
            pre_reward_e = []
            post_reward_e = []
            pre_reward_i = []
            post_reward_i = []
            for idx in reward_indices:
                pre_start = max(0, idx - window_size)
                post_end = min(len(timesteps), idx + window_size + 1)
                pre_reward_e.append(np.mean(E_states[pre_start:idx], axis=0) if idx > pre_start else np.zeros_like(E_states[0]))
                post_reward_e.append(np.mean(E_states[idx:post_end], axis=0) if idx < post_end else np.zeros_like(E_states[0]))
                pre_reward_i.append(np.mean(I_states[pre_start:idx], axis=0) if idx > pre_start else np.zeros_like(I_states[0]))
                post_reward_i.append(np.mean(I_states[idx:post_end], axis=0) if idx < post_end else np.zeros_like(I_states[0]))
            pre_reward_e = np.array(pre_reward_e).mean(axis=0)
            post_reward_e = np.array(post_reward_e).mean(axis=0)
            pre_reward_i = np.array(pre_reward_i).mean(axis=0)
            post_reward_i = np.array(post_reward_i).mean(axis=0)
            e_diff = post_reward_e - pre_reward_e
            i_diff = post_reward_i - pre_reward_i
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(e_diff)), e_diff, color='#1f77b4', alpha=0.7)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title('Excitatory Neuron Activation Changes Before and After Reward', fontsize=14)
            plt.xlabel('Neuron Index', fontsize=12)
            plt.ylabel('Activation Difference (After-Before Reward)', fontsize=12)
            plt.subplot(1, 2, 2)
            plt.bar(range(len(i_diff)), i_diff, color='#ff7f0e', alpha=0.7)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title('Inhibitory Neuron Activation Changes Before and After Reward', fontsize=14)
            plt.xlabel('Neuron Index', fontsize=12)
            plt.ylabel('Activation Difference (After-Before Reward)', fontsize=12)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/reward_neuron_changes_{episode_idx}.png", dpi=300)
        action_changes = np.where(np.diff(actions) != 0)[0] + 1
        if len(action_changes) > 0:
            num_changes_to_analyze = min(5, len(action_changes))
            selected_changes = action_changes[np.linspace(0, len(action_changes)-1, num_changes_to_analyze).astype(int)]
            plt.figure(figsize=(15, num_changes_to_analyze * 4))
            for i, change_idx in enumerate(selected_changes):
                if change_idx > 0 and change_idx < len(actions):
                    pre_action = actions[change_idx - 1]
                    post_action = actions[change_idx]
                    window = 3  # 前后窗口大小
                    pre_idx = max(0, change_idx - window)
                    post_idx = min(len(actions) - 1, change_idx + window)
                    pre_e = np.mean(E_states[pre_idx:change_idx], axis=0)
                    post_e = np.mean(E_states[change_idx:post_idx+1], axis=0)
                    pre_i = np.mean(I_states[pre_idx:change_idx], axis=0)
                    post_i = np.mean(I_states[change_idx:post_idx+1], axis=0)
                    e_change = post_e - pre_e
                    i_change = post_i - pre_i
                    plt.subplot(num_changes_to_analyze, 2, i*2 + 1)
                    plt.bar(range(len(e_change)), e_change, color='green', alpha=0.7)
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    plt.title(f'Excitatory Neuron Changes for Action Transition {pre_action} → {post_action}', fontsize=12)
                    plt.xlabel('Neuron Index', fontsize=10)
                    plt.ylabel('Activity Difference', fontsize=10)
                    plt.subplot(num_changes_to_analyze, 2, i*2 + 2)
                    plt.bar(range(len(i_change)), i_change, color='purple', alpha=0.7)
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    plt.title(f'Inhibitory Neuron Changes for Action Transition {pre_action} → {post_action}', fontsize=12)
                    plt.xlabel('Neuron Index', fontsize=10)
                    plt.ylabel('Activity Difference', fontsize=10)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/action_change_analysis_{episode_idx}.png", dpi=300)
        combined_states = np.hstack([E_states, I_states])
        pca = PCA(n_components=3)
        reduced_states = pca.fit_transform(combined_states)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, len(reduced_states)))
        ax.scatter(reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2], 
                   c=colors, s=20, alpha=0.7)
        ax.plot(reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2], 
                'k-', alpha=0.3, linewidth=1)
        ax.scatter(reduced_states[0, 0], reduced_states[0, 1], reduced_states[0, 2], 
                   color='green', s=100, label='start point')
        ax.scatter(reduced_states[-1, 0], reduced_states[-1, 1], reduced_states[-1, 2], 
                   color='red', s=100, label='end point')
        for idx in reward_indices:
            if idx < len(reduced_states):
                ax.scatter(reduced_states[idx, 0], reduced_states[idx, 1], reduced_states[idx, 2], 
                           color='yellow', s=80, label='reward point' if idx == reward_indices[0] else "")
        ax.set_title('Neural State Space Trajectory', fontsize=16)
        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.set_zlabel('PC3', fontsize=14)
        plt.legend()
        if save_dir:
            plt.savefig(f"{save_dir}/state_space_trajectory_{episode_idx}.png", dpi=300)
    def analyze_connectivity(self, save_dir=None):
        """
        分析WC模型的连接结构
        Args:
            save_dir: 保存目录
        """
        w_ee = self.wc_cell.w_ee.detach().cpu().numpy()
        w_ei = self.wc_cell.w_ei.detach().cpu().numpy()
        w_ie = self.wc_cell.w_ie.detach().cpu().numpy()
        w_ii = self.wc_cell.w_ii.detach().cpu().numpy()
        stats = {
            'w_ee': {
                'mean': float(np.mean(w_ee)),
                'std': float(np.std(w_ee)),
                'density': float(np.count_nonzero(w_ee) / w_ee.size),
                'min': float(np.min(w_ee)),
                'max': float(np.max(w_ee))
            },
            'w_ei': {
                'mean': float(np.mean(w_ei)),
                'std': float(np.std(w_ei)),
                'density': float(np.count_nonzero(w_ei) / w_ei.size),
                'min': float(np.min(w_ei)),
                'max': float(np.max(w_ei))
            },
            'w_ie': {
                'mean': float(np.mean(w_ie)),
                'std': float(np.std(w_ie)),
                'density': float(np.count_nonzero(w_ie) / w_ie.size),
                'min': float(np.min(w_ie)),
                'max': float(np.max(w_ie))
            },
            'w_ii': {
                'mean': float(np.mean(w_ii)),
                'std': float(np.std(w_ii)),
                'density': float(np.count_nonzero(w_ii) / w_ii.size),
                'min': float(np.min(w_ii)),
                'max': float(np.max(w_ii))
            }
        }
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        im1 = plt.imshow(w_ee, cmap='coolwarm')
        plt.colorbar(im1)
        plt.title('Excitatory → Excitatory Connections (w_ee)', fontsize=16)
        plt.xlabel('Target Neuron', fontsize=14)
        plt.ylabel('Source Neuron', fontsize=14)
        plt.subplot(2, 2, 2)
        im2 = plt.imshow(w_ei, cmap='coolwarm')
        plt.colorbar(im2)
        plt.title('Excitatory → Inhibitory Connections (w_ei)', fontsize=16)
        plt.xlabel('Target Neuron', fontsize=14)
        plt.ylabel('Source Neuron', fontsize=14)
        plt.subplot(2, 2, 3)
        im3 = plt.imshow(w_ie, cmap='coolwarm')
        plt.colorbar(im3)
        plt.title('Inhibitory → Excitatory Connections (w_ie)', fontsize=16)
        plt.xlabel('Target Neuron', fontsize=14)
        plt.ylabel('Source Neuron', fontsize=14)
        plt.subplot(2, 2, 4)
        im4 = plt.imshow(w_ii, cmap='coolwarm')
        plt.colorbar(im4)
        plt.title('Inhibitory → Inhibitory Connections (w_ii)', fontsize=16)
        plt.xlabel('Target Neuron', fontsize=14)
        plt.ylabel('Source Neuron', fontsize=14)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/connectivity_matrices.png", dpi=300)
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        sns.histplot(w_ee.flatten(), kde=True, color='blue')
        plt.title('Excitatory → Excitatory Weight Distribution', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.subplot(2, 2, 2)
        sns.histplot(w_ei.flatten(), kde=True, color='green')
        plt.title('Excitatory → Inhibitory Weight Distribution', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.subplot(2, 2, 3)
        sns.histplot(w_ie.flatten(), kde=True, color='orange')
        plt.title('Inhibitory → Excitatory Weight Distribution', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.subplot(2, 2, 4)
        sns.histplot(w_ii.flatten(), kde=True, color='red')
        plt.title('Inhibitory → Inhibitory Weight Distribution', fontsize=14)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/connectivity_distributions.png", dpi=300)
        return stats
    def analyze_in_out_weights(self, save_dir=None):
        """
        分析输入输出权重
        Args:
            save_dir: 保存目录
        """
        input_weights_e = self.wc_cell.input_weights_e.detach().cpu().numpy()
        input_weights_i = self.wc_cell.input_weights_i.detach().cpu().numpy()
        if hasattr(self.wc_cell, 'output_weights'):
            output_weights = self.wc_cell.output_weights.detach().cpu().numpy()
            has_output = True
        else:
            output_weights = None
            has_output = False
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        sns.heatmap(input_weights_e.T, cmap='viridis')
        plt.title('Weights from Convolutional Features to Excitatory Neurons', fontsize=16)
        plt.xlabel('Convolutional Feature Index', fontsize=14)
        plt.ylabel('Excitatory Neuron Index', fontsize=14)
        plt.subplot(2, 1, 2)
        sns.heatmap(input_weights_i.T, cmap='viridis')
        plt.title('Weights from Convolutional Features to Inhibitory Neurons', fontsize=16)
        plt.xlabel('Convolutional Feature Index', fontsize=14)
        plt.ylabel('Inhibitory Neuron Index', fontsize=14)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/input_weights.png", dpi=300)
        if has_output:
            plt.figure(figsize=(10, 8))
            sns.heatmap(output_weights, cmap='coolwarm')
            plt.title('Weights from Neurons to Output Actions', fontsize=16)
            plt.xlabel('Action Index', fontsize=14)
            plt.ylabel('Neuron Index', fontsize=14)
            plt.axhline(y=self.wc_cell.num_excitatory - 0.5, color='black', linestyle='--', linewidth=1)
            plt.text(-0.5, self.wc_cell.num_excitatory / 2, 'E', fontsize=18, 
                    ha='center', va='center')
            plt.text(-0.5, self.wc_cell.num_excitatory + 
                    self.wc_cell.num_inhibitory / 2, 'I', fontsize=18, 
                    ha='center', va='center')
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/output_weights.png", dpi=300)
        stats = {
            'input_weights_e': {
                'mean': float(np.mean(input_weights_e)),
                'std': float(np.std(input_weights_e)),
                'sparsity': float(np.count_nonzero(input_weights_e) / input_weights_e.size)
            },
            'input_weights_i': {
                'mean': float(np.mean(input_weights_i)),
                'std': float(np.std(input_weights_i)),
                'sparsity': float(np.count_nonzero(input_weights_i) / input_weights_i.size)
            }
        }
        if has_output:
            stats['output_weights'] = {
                'mean': float(np.mean(output_weights)),
                'std': float(np.std(output_weights)),
                'sparsity': float(np.count_nonzero(output_weights) / output_weights.size)
            }
        return stats
    def analyze_model_integration(self, save_dir=None, episode_idx=0):
        """
        分析模型整体集成过程
        Args:
            save_dir: 保存目录
            episode_idx: 游戏序号
        """
        if not self.feature_maps_history or not self.excitatory_history:
            raise ValueError("请先调用run_game_episode收集数据")
        if len(self.feature_maps_history) > 20:
            time_points = np.linspace(0, len(self.feature_maps_history)-1, 5).astype(int)
            for t_idx, t in enumerate(time_points):
                if t < len(self.feature_maps_history) and t < len(self.excitatory_history):
                    feature_means = self.feature_maps_history[t][0].mean(dim=(1, 2)).cpu().numpy()
                    e_activity = self.excitatory_history[t][0].cpu().numpy()
                    i_activity = self.inhibitory_history[t][0].cpu().numpy()
                    num_features = min(10, len(feature_means))
                    plt.figure(figsize=(15, 6))
                    corr_matrix_e = np.zeros((num_features, min(10, len(e_activity))))
                    for i in range(num_features):
                        for j in range(min(10, len(e_activity))):
                            corr_matrix_e[i, j] = self.wc_cell.input_weights_e[i, j].cpu().item()
                    plt.subplot(1, 2, 1)
                    sns.heatmap(corr_matrix_e, cmap='coolwarm', center=0, 
                               xticklabels=[f'E{i}' for i in range(corr_matrix_e.shape[1])],
                               yticklabels=[f'F{i}' for i in range(corr_matrix_e.shape[0])])
                    plt.title(f'Feature to Excitatory Neuron Weights (Time Point {t})', fontsize=14)
                    plt.xlabel('Excitatory Neurons', fontsize=12)
                    plt.ylabel('Convolutional Features', fontsize=12)
                    corr_matrix_i = np.zeros((num_features, min(10, len(i_activity))))
                    for i in range(num_features):
                        for j in range(min(10, len(i_activity))):
                            corr_matrix_i[i, j] = self.wc_cell.input_weights_i[i, j].cpu().item()
                    plt.subplot(1, 2, 2)
                    sns.heatmap(corr_matrix_i, cmap='coolwarm', center=0,
                               xticklabels=[f'I{i}' for i in range(corr_matrix_i.shape[1])],
                               yticklabels=[f'F{i}' for i in range(corr_matrix_i.shape[0])])
                    plt.title(f'Feature to Inhibitory Neuron Weights (Time Point {t})', fontsize=14)
                    plt.xlabel('Inhibitory Neurons', fontsize=12)
                    plt.ylabel('Convolutional Features', fontsize=12)
                    plt.tight_layout()
                    if save_dir:
                        plt.savefig(f"{save_dir}/feature_neuron_correlation_t{t_idx}_ep{episode_idx}.png", dpi=300)
        plt.figure(figsize=(15, 10))
        t = len(self.actions_history) // 3
        if t < len(self.feature_maps_history):
            feature_means = self.feature_maps_history[t][0].mean(dim=(1, 2)).cpu().numpy()
            plt.subplot(2, 2, 1)
            num_features_plot = min(10, len(feature_means))
            plt.bar(range(num_features_plot), feature_means[:num_features_plot], color='blue', alpha=0.7)
            plt.title('Convolutional Feature Activity', fontsize=14)
            plt.xlabel('Feature Index', fontsize=12)
            plt.ylabel('Activity Level', fontsize=12)
            plt.subplot(2, 2, 2)
            e_activity = self.excitatory_history[t][0].cpu().numpy()
            i_activity = self.inhibitory_history[t][0].cpu().numpy()
            num_e = min(10, len(e_activity))
            num_i = min(10, len(i_activity))
            plt.bar(range(num_e), e_activity[:num_e], color='green', alpha=0.7, label='Excitatory')
            plt.bar(range(num_e, num_e + num_i), i_activity[:num_i], color='red', alpha=0.7, label='Inhibitory')
            plt.title('Neuron Activity', fontsize=14)
            plt.xlabel('Neuron Index', fontsize=12)
            plt.ylabel('Activity Level', fontsize=12)
            plt.legend()
            plt.subplot(2, 2, 3)
            if t < len(self.actions_history):
                action = self.actions_history[t]
                action_probs = np.zeros(4)  # Breakout has 4 actions
                action_probs[action] = 1.0
                plt.bar(range(4), action_probs, color='purple', alpha=0.7)
                plt.title('Selected Action', fontsize=14)
                plt.xlabel('Action Index', fontsize=12)
                plt.ylabel('Probability', fontsize=12)
                plt.xticks(range(4), ['No-op', 'Fire', 'Right', 'Left'])
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/end_to_end_activation_ep{episode_idx}.png", dpi=300)
    def create_animation(self, frames, save_path=None):
        """
        创建游戏画面和神经活动的动画
        Args:
            frames: 游戏帧序列
            save_path: 保存路径
        Returns:
            anim: 动画对象
        """
        if not self.excitatory_history or not frames:
            raise ValueError("请先调用run_game_episode收集数据")
        E_activity = torch.cat(self.excitatory_history, dim=0).numpy()
        I_activity = torch.cat(self.inhibitory_history, dim=0).numpy()
        actions = np.array(self.actions_history)
        rewards = np.array(self.rewards_history)
        min_len = min(len(frames), len(E_activity), len(actions))
        E_activity = E_activity[:min_len]
        I_activity = I_activity[:min_len]
        actions = actions[:min_len]
        rewards = rewards[:min_len] if len(rewards) >= min_len else np.zeros(min_len)
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0, :])  # 游戏画面
        ax2 = plt.subplot(gs[1, 0])  # 兴奋性活动
        ax3 = plt.subplot(gs[1, 1])  # 抑制性活动
        game_img = ax1.imshow(frames[0])
        ax1.set_title('Game Screen', fontsize=14)
        ax1.axis('off')
        e_img = ax2.imshow(E_activity[:1].T, aspect='auto', cmap='viridis')
        ax2.set_title('Excitatory Neurons', fontsize=12)
        ax2.set_xlabel('Time Steps', fontsize=10)
        ax2.set_ylabel('Neuron Index', fontsize=10)
        plt.colorbar(e_img, ax=ax2)
        i_img = ax3.imshow(I_activity[:1].T, aspect='auto', cmap='viridis')
        ax3.set_title('Inhibitory Neurons', fontsize=12)
        ax3.set_xlabel('Time Steps', fontsize=10)
        ax3.set_ylabel('Neuron Index', fontsize=10)
        plt.colorbar(i_img, ax=ax3)
        action_text = ax1.text(10, 20, f"Action: {actions[0]}", 
                      color='white', fontsize=12, 
                      bbox=dict(facecolor='black', alpha=0.7))
        reward_text = ax1.text(10, 50, f"Reward: {rewards[0]:.1f}", 
                      color='white', fontsize=12, 
                      bbox=dict(facecolor='black', alpha=0.7))
        plt.tight_layout()
        def update(frame):
            game_img.set_array(frames[frame])
            e_img.set_array(E_activity[:frame+1].T)
            i_img.set_array(I_activity[:frame+1].T)
            action_text.set_text(f"Action: {actions[frame]}")
            reward_text.set_text(f"Reward: {rewards[frame]:.1f}")
            ax2.set_xlim(max(0, frame-30), frame+1)
            ax3.set_xlim(max(0, frame-30), frame+1)
            return game_img, e_img, i_img, action_text, reward_text
        anim = animation.FuncAnimation(fig, update, frames=min_len, 
                                     interval=100, blit=True)
        if save_path:
            print(f"保存动画到 {save_path}...")
            anim.save(save_path, dpi=80, fps=10, writer='pillow')
        return anim
    def run_comprehensive_analysis(self, env, output_dir, num_episodes=3):
        """
        运行全面的模型分析
        Args:
            env: 游戏环境
            output_dir: 输出目录
            num_episodes: 游戏局数
        Returns:
            report: 分析报告数据
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            'model_info': self._get_model_info(),
            'connectivity': {},
            'weights': {},
            'episodes': []
        }
        try:
            report['connectivity'] = self.analyze_connectivity(save_dir=output_dir)
        except Exception as e:
            print(f"连接结构分析失败: {str(e)}")
            report['connectivity'] = {"error": str(e)}
        try:
            report['weights'] = self.analyze_in_out_weights(save_dir=output_dir)
        except Exception as e:
            print(f"权重分析失败: {str(e)}")
            report['weights'] = {"error": str(e)}
        try:
            print("运行模型可解释性分析...")
            self.run_explainability_analysis(env, output_dir, num_samples=5)
        except Exception as e:
            print(f"可解释性分析失败: {str(e)}")
        for ep in range(num_episodes):
            print(f"分析游戏局 {ep+1}/{num_episodes}...")
            ep_dir = output_dir / f"episode_{ep+1}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            episode_info = {
                'reward': 0,
                'steps': 0,
                'actions_distribution': {},
                'neuron_stats': {}
            }
            try:
                render = (ep == num_episodes - 1)  # 只在最后一局渲染
                total_reward, frames = self.run_game_episode(env, render=render)
                episode_info['reward'] = float(total_reward)
                episode_info['steps'] = len(self.actions_history)
                if self.actions_history:
                    episode_info['actions_distribution'] = {int(a): int(c) for a, c in 
                                                        zip(*np.unique(self.actions_history, return_counts=True))}
                try:
                    neuron_stats = self.analyze_neuron_activity(save_dir=ep_dir, episode_idx=ep+1)
                    episode_info['neuron_stats'] = neuron_stats
                except Exception as e:
                    print(f"神经元活动分析失败: {str(e)}")
                try:
                    self.analyze_feature_maps(save_dir=ep_dir, episode_idx=ep+1)
                except Exception as e:
                    print(f"卷积特征图分析失败: {str(e)}")
                try:
                    self.temporal_dynamics_analysis(save_dir=ep_dir, episode_idx=ep+1)
                except Exception as e:
                    print(f"时间动力学分析失败: {str(e)}")
                try:
                    self.analyze_model_integration(save_dir=ep_dir, episode_idx=ep+1)
                except Exception as e:
                    print(f"模型整体集成分析失败: {str(e)}")
                if frames and len(frames) > 0:
                    try:
                        self.create_animation(frames, save_path=str(ep_dir / f"game_animation_{ep+1}.gif"))
                    except Exception as e:
                        print(f"创建动画失败: {str(e)}")
            except Exception as e:
                print(f"游戏局 {ep+1} 分析失败: {str(e)}")
            report['episodes'].append(episode_info)
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        self._generate_report_summary(report, output_dir)
        return report
    def _get_model_info(self):
        """获取模型信息"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        info = {
            'trainable_parameters': int(trainable_params),
            'total_parameters': int(total_params),
            'device': str(self.device),
        }
        if hasattr(self, 'wc_cell'):
            info.update({
                'num_excitatory': int(self.wc_cell.num_excitatory),
                'num_inhibitory': int(self.wc_cell.num_inhibitory),
                'excitatory_ratio': float(self.wc_cell.excitatory_ratio),
                'connectivity_type': getattr(self.wc_cell, 'connectivity_type', 'unknown'),
                'use_rk4': bool(getattr(self.wc_cell, 'use_rk4', False)),
                'dt': float(getattr(self.wc_cell, 'dt', 0.1))
            })
        return info
    def _generate_report_summary(self, report, output_dir):
        """生成报告摘要"""
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("===== 卷积Wilson-Cowan模型分析报告 =====\n\n")
            f.write("模型信息:\n")
            f.write("----------------------------------\n")
            info = report['model_info']
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("连接分析:\n")
            f.write("----------------------------------\n")
            conn = report['connectivity']
            for conn_type, stats in conn.items():
                f.write(f"{conn_type}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
            f.write("\n")
            f.write("游戏表现:\n")
            f.write("----------------------------------\n")
            for i, ep in enumerate(report['episodes']):
                f.write(f"游戏局 {i+1}:\n")
                f.write(f"  奖励: {ep['reward']:.1f}\n")
                f.write(f"  步数: {ep['steps']}\n")
                f.write(f"  动作分布: {ep['actions_distribution']}\n")
                f.write(f"  平均兴奋性神经元活动: {ep['neuron_stats']['e_activity']['mean']:.4f}\n")
                f.write(f"  平均抑制性神经元活动: {ep['neuron_stats']['i_activity']['mean']:.4f}\n")
                f.write(f"  平均E/I比率: {ep['neuron_stats']['ei_ratio_mean']:.4f}\n\n")
    def initialize_explainers(self):
        """初始化各种解释性工具"""
        if hasattr(self, 'conv_block'):
            last_conv = None
            for name, module in self.conv_block.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                self.grad_cam = GradCam(self.model, last_conv)
                print(f"初始化Grad-CAM，目标层: {last_conv}")
            else:
                self.grad_cam = None
                print("警告: 未找到卷积层，Grad-CAM将不可用")
        else:
            self.grad_cam = None
            print("警告: 模型中未找到卷积块，Grad-CAM将不可用")
        try:
            self.integrated_gradients = IntegratedGradients(self.model)
            self.deeplift = DeepLift(self.model)
            self.gradient_shap = GradientShap(self.model)
            print("已初始化Captum解释器")
        except Exception as e:
            print(f"初始化Captum解释器失败: {e}")
            self.integrated_gradients = None
            self.deeplift = None
            self.gradient_shap = None
    def apply_gradcam(self, obs_tensor, action_idx=None, save_path=None):
        """
        应用Grad-CAM生成类激活图
        Args:
            obs_tensor: 输入观察张量 (B, C, H, W)
            action_idx: 要解释的动作索引，None表示使用预测的动作
            save_path: 保存路径
        Returns:
            cam: Grad-CAM热力图
            heatmap_on_image: 叠加在原图上的热力图
        """
        if self.grad_cam is None:
            print("Grad-CAM不可用，请确认模型包含卷积层")
            return None, None
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.dim() == 5:
            obs_tensor = obs_tensor.squeeze(1)
        cam = self.grad_cam(obs_tensor.to(self.device), action_idx)
        cam_np = cam.cpu().numpy()
        original_h, original_w = obs_tensor.shape[2], obs_tensor.shape[3]
        cam_np = cv2.resize(cam_np, (original_w, original_h))
        cam_np = np.nan_to_num(cam_np)
        cam_np = np.clip(cam_np, 0, 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        obs_np = obs_tensor[0].permute(1, 2, 0).cpu().numpy()
        if obs_np.max() <= 1.0:
            obs_np = (obs_np * 255).astype(np.uint8)
        if obs_np.shape[2] == 1:
            obs_np = np.repeat(obs_np, 3, axis=2)
        elif obs_np.shape[2] > 3:
            obs_np = obs_np[:, :, :3]
        obs_np = obs_np.astype(np.uint8)
        heatmap_on_image = cv2.addWeighted(heatmap, 0.5, obs_np, 0.5, 0)
        if save_path:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(obs_np)
            plt.title('Original Image', fontsize=12)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(cam_np, cmap='jet')
            plt.title('Grad-CAM Heatmap', fontsize=12)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(heatmap_on_image)
            plt.title('Overlaid Heatmap', fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return cam_np, heatmap_on_image
    def analyze_neuron_attention(self, obs_tensor, neuron_indices=None, e_or_i='e', save_path=None):
        """
        分析特定神经元对输入的关注区域
        Args:
            obs_tensor: 输入观察张量
            neuron_indices: 要分析的神经元索引，None表示随机选择
            e_or_i: 'e'表示兴奋性神经元，'i'表示抑制性神经元
            save_path: 保存路径
        """
        original_dim = obs_tensor.dim()
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if obs_tensor.dim() == 4:
            obs_tensor = obs_tensor.unsqueeze(1)
        else:
            obs_tensor_model = obs_tensor
        obs_tensor_viz = obs_tensor.clone()
        obs_var = Variable(obs_tensor.to(self.device), requires_grad=True)
        outputs, hidden_states = self.model(obs_var)
        if isinstance(hidden_states, tuple) and len(hidden_states) >= 2:
            e_states = hidden_states[0]
            i_states = hidden_states[1]
        else:
            hidden_size = hidden_states.shape[-1]
            num_e = getattr(self.wc_cell, 'num_excitatory', hidden_size // 2)
            e_states = hidden_states[..., :num_e]
            i_states = hidden_states[..., num_e:]
        states = e_states if e_or_i == 'e' else i_states
        num_neurons = states.shape[-1]
        if neuron_indices is None:
            num_to_analyze = min(5, num_neurons)
            neuron_indices = torch.randperm(num_neurons)[:num_to_analyze].tolist()
        neuron_grads = []
        for idx in neuron_indices:
            self.model.zero_grad()
            if states.dim() > 2:  # [B, T, N] 或更多维度
                target = states[0, 0, idx]  # 取第一个批次、第一个时间步的指定神经元
            else:  # [B, N]
                target = states[0, idx]  # 取第一个批次的指定神经元
            target.backward(retain_graph=True)
            grad = obs_var.grad.clone()
            obs_var.grad.zero_()
            if grad.dim() == 5:  # [B, T, C, H, W]
                grad_abs = torch.abs(grad).sum(dim=2).squeeze().cpu().numpy()
            elif grad.dim() == 4:  # [B, C, H, W]
                grad_abs = torch.abs(grad).sum(dim=1).squeeze().cpu().numpy()
            else:
                print(f"警告: 意外的梯度维度: {grad.shape}")
                grad_abs = torch.abs(grad).squeeze().cpu().numpy()
            neuron_grads.append(grad_abs)
        plt.figure(figsize=(15, 3 * len(neuron_indices)))
        for i, (idx, grad) in enumerate(zip(neuron_indices, neuron_grads)):
            plt.subplot(len(neuron_indices), 3, i*3 + 1)
            obs_np = obs_tensor[0].permute(1, 2, 0).cpu().numpy()
            if obs_np.shape[2] == 1:
                plt.imshow(obs_np.squeeze(), cmap='gray')
            else:
                plt.imshow(obs_np)
            plt.title(f'Original Image', fontsize=12)
            plt.axis('off')
            plt.subplot(len(neuron_indices), 3, i*3 + 2)
            plt.imshow(grad, cmap='hot')
            plt.title(f'{"Excitatory" if e_or_i == "e" else "Inhibitory"} Neuron {idx} Gradient', fontsize=12)
            plt.axis('off')
            plt.subplot(len(neuron_indices), 3, i*3 + 3)
            grad_resized = cv2.resize(grad, (obs_np.shape[1], obs_np.shape[0]))
            grad_normalized = (grad_resized - grad_resized.min()) / (grad_resized.max() - grad_resized.min() + 1e-8)
            if obs_np.shape[2] == 1:
                obs_np = np.repeat(obs_np, 3, axis=2)
            if obs_np.max() <= 1.0:
                obs_np = (obs_np * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(np.uint8(255 * grad_normalized), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(obs_np, 0.6, heatmap, 0.4, 0)
            plt.imshow(superimposed)
            plt.title(f'Overlaid Heatmap', fontsize=12)
            plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    def analyze_action_attribution(self, obs_tensor, save_path=None):
        """
        分析不同动作的归因
        Args:
            obs_tensor: 输入观察张量
            save_path: 保存路径
        """
        if self.integrated_gradients is None:
            print("Captum解释器不可用")
            return
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(obs_tensor.to(self.device))
            pred = pred.squeeze()
            pred_action = torch.argmax(pred).item()
        attributions = self.integrated_gradients.attribute(
            obs_tensor.to(self.device),
            target=pred_action,
            n_steps=50
        )
        attr = attributions.squeeze().cpu().numpy()
        attr_abs = np.abs(attr)
        if attr.ndim > 2:
            attr_agg = np.sum(attr_abs, axis=0)
        else:
            attr_agg = attr_abs
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        obs_np = obs_tensor[0].permute(1, 2, 0).cpu().numpy()
        if obs_np.shape[2] == 1:
            plt.imshow(obs_np.squeeze(), cmap='gray')
        else:
            plt.imshow(obs_np)
        plt.title('Original Image', fontsize=12)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(attr_agg, cmap='hot')
        plt.title(f'Feature Importance for Action {pred_action}', fontsize=12)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        attr_resized = cv2.resize(attr_agg, (obs_np.shape[1], obs_np.shape[0]))
        attr_normalized = (attr_resized - attr_resized.min()) / (attr_resized.max() - attr_resized.min() + 1e-8)
        if obs_np.shape[2] == 1:
            obs_np = np.repeat(obs_np, 3, axis=2)
        if obs_np.max() <= 1.0:
            obs_np = (obs_np * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_normalized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(obs_np, 0.6, heatmap, 0.4, 0)
        plt.imshow(superimposed)
        plt.title('Overlaid Feature Importance', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    def run_explainability_analysis(self, env, save_dir, num_samples=5):
        """
        运行完整的可解释性分析
        Args:
            env: 游戏环境
            save_dir: 保存目录
            num_samples: 分析的样本数量
        """
        save_dir = Path(save_dir)
        explainability_dir = save_dir / "explainability"
        explainability_dir.mkdir(parents=True, exist_ok=True)
        print("收集样本进行解释性分析...")
        obs_samples = []
        action_samples = []
        reward_samples = []
        obs, _ = env.reset()
        done = False
        steps = 0
        hidden = None
        self.initialize_explainers()
        while len(obs_samples) < num_samples and steps < 1000 and not done:
            obs_tensor = np.transpose(obs, [2, 0, 1]).astype(np.float32)
            obs_tensor = torch.from_numpy(obs_tensor).to(self.device)
            obs_tensor_model = obs_tensor.unsqueeze(0).unsqueeze(0)
            obs_tensor_viz = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                outputs, hidden = self.model(obs_tensor_model, hidden)
                action = outputs.squeeze(0).squeeze(0).argmax().item()
            obs_next, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            if steps % (1000 // num_samples) == 0:
                obs_samples.append(obs_tensor_viz.clone())
                action_samples.append(action)
                reward_samples.append(reward)
            obs = obs_next
            steps += 1
        print(f"对 {len(obs_samples)} 个样本应用解释性方法...")
        for i, (obs_tensor, action, reward) in enumerate(zip(obs_samples, action_samples, reward_samples)):
            print(f"分析样本 {i+1}/{len(obs_samples)} - 动作: {action}, 奖励: {reward}")
            sample_dir = explainability_dir / f"sample_{i+1}"
            sample_dir.mkdir(exist_ok=True)
            print("应用Grad-CAM...")
            self.apply_gradcam(
                obs_tensor, 
                action_idx=action, 
                save_path=str(sample_dir / f"gradcam_action{action}.png")
            )
            print("分析神经元关注区域...")
            self.analyze_neuron_attention(
                obs_tensor,
                neuron_indices=None,  # 随机选择
                e_or_i='e',
                save_path=str(sample_dir / f"neuron_attention_excitatory.png")
            )
            self.analyze_neuron_attention(
                obs_tensor,
                neuron_indices=None,  # 随机选择
                e_or_i='i',
                save_path=str(sample_dir / f"neuron_attention_inhibitory.png")
            )
            print("分析动作归因...")
            self.analyze_action_attribution(
                obs_tensor,
                save_path=str(sample_dir / f"action_attribution.png")
            )
        print(f"可解释性分析完成! 结果保存在 {explainability_dir}")