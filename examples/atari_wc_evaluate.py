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
from ConvWCEvaluator import *
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
# 自定义颜色映射
custom_cmap1 = LinearSegmentedColormap.from_list("custom_viridis_red", 
                                              [(0, "#440154"), (0.5, "#21918c"), (1, "#ff4040")])
custom_cmap2 = LinearSegmentedColormap.from_list("custom_purple_green", 
                                              [(0, "#9c179e"), (0.5, "#f7d03c"), (1, "#21908d")])
def load_model(model_path, device='cpu'):
    """
    加载保存的模型
    Args:
        model_path: 模型文件路径
        device: 运行设备
    Returns:
        model: 加载的模型
        checkpoint: 模型检查点数据
    """
    print(f"从 {model_path} 加载模型...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 分析检查点内容，推断模型配置
        if 'model_type' in checkpoint and 'connection_type' in checkpoint and 'hidden_size' in checkpoint:
            model_type = checkpoint['model_type']
            connection_type = checkpoint['connection_type']
            hidden_size = checkpoint['hidden_size']
        else:
            # 默认配置
            model_type = 'wc'
            connection_type = 'small-world'
            hidden_size = 64
        # 从检查点中推断或设置是否使用ODE
        use_ode = checkpoint.get('use_ode', True)
        # 创建模型
        model = create_model(
            model_type=model_type.lower(),
            n_actions=4,  # Breakout游戏的动作数量
            hidden_size=hidden_size,
            connection_type=connection_type.lower(),
            use_ode=use_ode
        )
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"成功加载模型: {model_type} ({connection_type}), hidden_size={hidden_size}")
        return model, checkpoint
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        # 尝试灵活加载
        print("尝试灵活加载模型...")
        checkpoint = torch.load(model_path, map_location=device)
        # 检查状态字典格式
        if 'model_state_dict' in checkpoint:
            # 从状态字典推断模型特征
            state_dict = checkpoint['model_state_dict']
            # 查找特征
            wc_layers = [k for k in state_dict.keys() if 'wc_cell' in k]
            conv_layers = [k for k in state_dict.keys() if 'conv' in k]
            # 有WC层和卷积层
            if wc_layers and conv_layers:
                print("检测到卷积WilsonCowan模型")
                # 推断隐藏层大小
                if any('w_ee' in k for k in wc_layers):
                    for key in wc_layers:
                        if 'w_ee' in key:
                            hidden_size = state_dict[key].shape[0]
                            break
                else:
                    hidden_size = 64  # 默认值
                # 推断连接类型
                connection_type = 'small-world'  # 默认值
                model_type = 'wc'
                # 创建模型
                model = create_model(
                    model_type=model_type,
                    n_actions=4,  # Breakout游戏的动作数量
                    hidden_size=hidden_size,
                    connection_type=connection_type,
                    use_ode=True
                )
                # 灵活加载参数
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                print(f"成功加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                model.to(device)
                model.eval()
                return model, checkpoint
        raise ValueError(f"无法识别的模型格式，请提供有效的卷积WilsonCowan模型")

def main():
    parser = argparse.ArgumentParser(description='卷积Wilson-Cowan模型评估工具')
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型文件路径')
    parser.add_argument('--output_dir', type=str, default='./conv_wc_analysis',
                      help='输出目录')
    parser.add_argument('--num_episodes', type=int, default=3,
                      help='评估的游戏局数')
    parser.add_argument('--cuda', action='store_true',
                      help='使用CUDA加速')
    parser.add_argument('--game', type=str, default='Breakout',
                      help='游戏名称 (默认: Breakout)')
    args = parser.parse_args()
    # 设置设备
    device = torch.device('cuda:2' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # 加载模型
    model, checkpoint = load_model(args.model_path, device)
    # 创建游戏环境
    env = gym.make(f"ALE/{args.game}-v5", render_mode="rgb_array")
    env = wrap_deepmind(env)
    # 创建评估器
    evaluator = ConvWCEvaluator(model, device)
    # 运行全面分析
    print("开始全面分析...")
    report = evaluator.run_comprehensive_analysis(env, output_dir, args.num_episodes)
    # evaluator.run_explainability_analysis(env, output_dir, args.num_episodes)
    print(f"分析完成! 结果保存在 {output_dir}")
    
if __name__ == "__main__":
    main()