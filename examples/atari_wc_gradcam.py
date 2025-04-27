import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import gymnasium as gym
from tqdm import tqdm
import cv2
import ale_py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from atari_cell import Conv3WilsonCowan_FC, Conv3WilsonCowan_Random, Conv3WilsonCowan_SmallWorld, Conv3WilsonCowan_Hierarchical
from ncps.datasets.torch import AtariCloningDataset

# 设置 Seaborn 样式
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({'font.size': 12})

class WilsonCowanGradCAM:
    """
    GradCAM 实现，专为 Wilson-Cowan 模型设计
    仅对卷积层部分进行可视化
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # 如果没有指定目标层，自动选择最后一个卷积层
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
            print(f"自动选择目标层: {target_layer}")
        
        # 从模型中获取目标层
        self.target_layer = self._get_target_layer(model, target_layer)
        print(f"使用目标层: {target_layer}, 类型: {type(self.target_layer).__name__}")
        
        # 存储激活和梯度
        self.activations = None
        self.gradients = None
        
        # 注册钩子函数
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
        
        # 获取设备
        self.device = next(model.parameters()).device
        
    def _find_last_conv_layer(self, model):
        """自动查找模型中的最后一个卷积层"""
        last_conv = None
        last_conv_name = ""
        
        def search_conv(model, prefix=""):
            nonlocal last_conv, last_conv_name
            for name, module in model.named_children():
                current_path = f"{prefix}.{name}" if prefix else name
                if isinstance(module, nn.Conv2d):
                    last_conv = module
                    last_conv_name = current_path
                search_conv(module, current_path)
        
        search_conv(model)
        return last_conv_name
    
    def _get_target_layer(self, model, target_layer_name):
        """获取指定名称的模型层"""
        # 如果传入的是字符串，解析为模型中对应的层
        if isinstance(target_layer_name, str):
            parts = target_layer_name.split('.')
            layer = model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        # 如果直接传入层对象，则直接返回
        return target_layer_name

    def _save_activation(self, module, input, output):
        """保存目标层的前向传播激活值"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """保存目标层的梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """生成 GradCAM 热力图，仅关注卷积特征"""
        # 清除旧的梯度
        self.model.zero_grad()
        
        # 保存原始输入形状，以便后续处理
        original_shape = input_tensor.shape
        
        # 前向传播
        output, _ = self.model(input_tensor)
        
        # 如果未指定目标类别，则使用预测的最高概率类别
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        # 确保 target_class 是一个张量
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class]).to(self.device)
        
        # 创建与输出相同形状的 one-hot 向量
        one_hot = torch.zeros_like(output)
        
        # 根据模型输出的维度，适应不同的索引方式
        if len(output.shape) == 3:  # [batch, seq_len, num_classes]
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    if i < len(target_class):
                        tc = target_class[i].item()
                        one_hot[i, j, tc] = 1
        elif len(output.shape) == 2:  # [batch, num_classes]
            for i in range(output.shape[0]):
                if i < len(target_class):
                    tc = target_class[i].item()
                    one_hot[i, tc] = 1
        else:
            print(f"警告: 不支持的输出形状 {output.shape}")
            # 为不支持的形状创建一个哑的 CAM
            dummy_cam = np.ones((original_shape[0], 84, 84))
            return dummy_cam
        
        # 反向传播
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 检查梯度和激活
        if self.gradients is None or self.activations is None:
            print("警告: 梯度或激活为空，请确保选择了正确的卷积层")
            dummy_cam = np.ones((original_shape[0], 84, 84))  # 标准的 Atari 预处理尺寸
            return dummy_cam
        
        # 获取每个通道的权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 加权激活图
        cam = torch.sum(weights * self.activations, dim=1)
        
        # 应用 ReLU 激活
        cam = torch.clamp(cam, min=0)
        
        # 标准化处理
        batch_size = cam.shape[0]
        normalized_cam = []
        
        for i in range(batch_size):
            cam_i = cam[i].detach().cpu().numpy()
            cam_i = cv2.resize(cam_i, (84, 84))  # 调整到标准 Atari 尺寸
            
            # 归一化
            if np.max(cam_i) - np.min(cam_i) > 1e-8:
                cam_i = (cam_i - np.min(cam_i)) / (np.max(cam_i) - np.min(cam_i))
            else:
                cam_i = np.zeros_like(cam_i)
            
            normalized_cam.append(cam_i)
        
        return np.array(normalized_cam)

    def overlay_cam(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """将 CAM 热力图与原始图像叠加显示"""
        # 确保输入是 numpy 数组
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        
        # 处理通道顺序：将 PyTorch 的 (C, H, W) 转为 (H, W, C)
        if len(image.shape) == 3 and (image.shape[0] == 3 or image.shape[0] == 4):
            image = np.transpose(image, (1, 2, 0))
        
        # 确保图像有 3 个通道以便于可视化
        if len(image.shape) == 2:  # 如果是单通道灰度图
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        # 确保 CAM 不包含 NaN 或无穷值
        cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保 CAM 值域在 [0,1] 范围内
        cam = np.clip(cam, 0, 1)
        
        # 将图像转换为 uint8 类型
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 调整 CAM 的大小以匹配图像大小
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # 将 CAM 转换为热力图
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
        
        # 将热力图叠加到原始图像上
        try:
            overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        except cv2.error as e:
            print(f"叠加错误: image shape {image.shape}, heatmap shape {heatmap.shape}")
            print(f"尝试调整尺寸和通道...")
            
            # 确保图像是 3 通道的
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # 确保热力图与图像尺寸一致
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            
            try:
                overlaid = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
            except cv2.error:
                print("仍然失败，返回原始图像")
                return image, heatmap
        
        return overlaid, heatmap

def list_all_layers(model):
    """列出模型中的所有层"""
    layers = []
    
    def traverse_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layers.append((full_name, type(child).__name__))
            traverse_layers(child, full_name)
    
    traverse_layers(model)
    return layers

def load_model(model_path, model_type, connection_type, hidden_size, n_actions=4, use_ode=False, dt=0.1):
    """加载预训练的 WC 模型"""
    if connection_type == 'fc':
        model = Conv3WilsonCowan_FC(n_actions, hidden_size, dt=dt, use_rk4=use_ode)
    elif connection_type == 'small-world':
        model = Conv3WilsonCowan_SmallWorld(n_actions, hidden_size, sparsity_level=0.5, dt=dt, use_rk4=use_ode)
    elif connection_type == 'hierarchical':
        model = Conv3WilsonCowan_Hierarchical(n_actions, hidden_size, sparsity_level=0.5, dt=dt, use_rk4=use_ode)
    else:  # random/ncp
        model = Conv3WilsonCowan_Random(n_actions, hidden_size, sparsity_level=0.5, dt=dt, use_rk4=use_ode)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def analyze_game_frame(model, grad_cam, env, device, output_dir):
    """对游戏帧进行 GradCAM 分析"""
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化环境和状态
    obs, info = env.reset()
    hx = None  # RNN 隐藏状态
    done = False
    step = 0
    total_reward = 0
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 收集一些游戏帧进行分析
    frames_to_analyze = []
    
    # 收集前 10 帧或直到游戏结束
    max_frames = 10
    collected = 0
    
    with torch.no_grad():
        while not done and collected < max_frames:
            # 转换观察为 PyTorch 张量
            obs_tensor = np.transpose(obs, [2, 0, 1]).astype(np.float32)
            obs_tensor = torch.from_numpy(obs_tensor).unsqueeze(0).unsqueeze(0).to(device)
            
            # 保存当前帧进行 GradCAM 分析
            frames_to_analyze.append((obs.copy(), obs_tensor.clone(), hx))
            collected += 1
            
            # 前向传播获取动作
            with torch.no_grad():
                pred, hx = model(obs_tensor, hx)
                action = pred.squeeze(0).squeeze(0).argmax().item()
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            step += 1
    
    # 对收集的帧进行 GradCAM 分析
    for i, (raw_frame, frame_tensor, current_hx) in enumerate(frames_to_analyze):
        if i >= 8:  # 限制在图中显示的帧数
            break
            
        # 运行 GradCAM
        cam = grad_cam.generate_cam(frame_tensor, target_class=None)
        
        # 将 CAM 与原始图像叠加
        overlaid, heatmap = grad_cam.overlay_cam(raw_frame, cam[0], alpha=0.5)
        
        # 获取预测的动作
        with torch.no_grad():
            pred, _ = model(frame_tensor, current_hx)
            action = pred.squeeze(0).squeeze(0).argmax().item()
            action_probs = torch.softmax(pred.squeeze(0).squeeze(0), dim=0).cpu().numpy()
        
        # 在子图中显示
        axes[i].imshow(overlaid)
        action_text = f"Action: {action_names[action]}\n"
        action_text += "\n".join([f"{name}: {prob:.3f}" for name, prob in zip(action_names, action_probs)])
        axes[i].set_title(action_text, fontsize=10)
        axes[i].axis('off')
    
    # 关闭未使用的子图
    for i in range(len(frames_to_analyze), 8):
        axes[i].axis('off')
    
    # 保存分析结果
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 为每一帧创建单独的详细分析图
    for i, (raw_frame, frame_tensor, current_hx) in enumerate(frames_to_analyze):
        # 运行 GradCAM
        cam = grad_cam.generate_cam(frame_tensor, target_class=None)
        if frame_tensor.shape[1] > 1:  # 如果有多个通道
            input_frame = frame_tensor[0, -1].cpu().numpy()  # 取最后一个通道
        else:
            input_frame = frame_tensor[0, 0].cpu().numpy()
        # 将 CAM 与原始图像叠加
        overlaid, heatmap = grad_cam.overlay_cam(input_frame, cam[0], alpha=0.5)
        
        # 获取预测的动作
        with torch.no_grad():
            pred, _ = model(frame_tensor, current_hx)
            action = pred.squeeze(0).squeeze(0).argmax().item()
            action_probs = torch.softmax(pred.squeeze(0).squeeze(0), dim=0).cpu().numpy()
        
        # 创建更详细的单帧分析
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(raw_frame)
        axes[0].set_title("Original Frame", fontsize=12)
        axes[0].axis('off')
        
        # Grad-CAM 热力图
        axes[1].imshow(heatmap)
        axes[1].set_title("GradCAM Heatmap", fontsize=12)
        axes[1].axis('off')
        
        # 叠加图像
        axes[2].imshow(overlaid)
        axes[2].set_title(f"Overlaid: Action = {action_names[action]}", fontsize=12)
        axes[2].axis('off')
        
        # 添加动作概率信息
        plt.figtext(0.5, 0.01, f"Action probabilities: " + ", ".join([f"{name}: {prob:.3f}" for name, prob in zip(action_names, action_probs)]), 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.savefig(output_dir / f"frame_{i}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return frames_to_analyze

def analyze_validation_samples(model, grad_cam, val_dataset, device, output_dir, num_samples=8):
    """对验证集样本进行 GradCAM 分析"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器获取单个样本
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=0
    )
    
    # 动作名称
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    
    # 分析多个验证样本
    analyzed_count = 0
    
    for inputs, labels in val_loader:
        if analyzed_count >= num_samples:
            break
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 调试输出
        print(f"输入形状: {inputs.shape}, 标签形状: {labels.shape}")
        
        # 运行 GradCAM，根据标签形状调整处理
        if len(labels.shape) > 1:
            target_class = labels[:, 0]  # 取第一个时间步的标签
        else:
            target_class = labels
            
        cam = grad_cam.generate_cam(inputs, target_class=target_class)
        
        # 将输入转换为可视化格式
        input_frame = inputs[0, -1].cpu().numpy()  # 只取第一个时间步
        
        # 将 CAM 与原始图像叠加
        overlaid, heatmap = grad_cam.overlay_cam(input_frame, cam[0], alpha=0.5)
        
        # 获取模型预测
        with torch.no_grad():
            outputs, _ = model(inputs)
            
            # 处理不同形状的输出
            if len(outputs.shape) == 3:  # [batch, seq, classes]
                pred_action = outputs[0, 0].argmax().item()
                action_probs = torch.softmax(outputs[0, 0], dim=0).cpu().numpy()
                true_action = labels[0, 0].item()
            else:  # [batch, classes]
                pred_action = outputs[0].argmax().item()
                action_probs = torch.softmax(outputs[0], dim=0).cpu().numpy()
                true_action = labels[0].item()
        
        # 创建详细的分析图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(np.transpose(input_frame, (1, 2, 0)))
        axes[0].set_title("Original Frame", fontsize=12)
        axes[0].axis('off')
        
        # Grad-CAM 热力图
        axes[1].imshow(heatmap)
        axes[1].set_title("GradCAM Heatmap", fontsize=12)
        axes[1].axis('off')
        
        # 叠加图像
        axes[2].imshow(overlaid)
        title = f"Pred: {action_names[pred_action]} (True: {action_names[true_action]})"
        color = "green" if pred_action == true_action else "red"
        axes[2].set_title(title, fontsize=12, color=color)
        axes[2].axis('off')
        
        # 添加动作概率信息
        plt.figtext(0.5, 0.01, f"Action probabilities: " + ", ".join([f"{name}: {prob:.3f}" for name, prob in zip(action_names, action_probs)]), 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.savefig(output_dir / f"val_sample_{analyzed_count}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        analyzed_count += 1
    
def main():
    parser = argparse.ArgumentParser(description='GradCAM Visualization for Wilson-Cowan Atari Models')
    parser.add_argument('--model_path', type=str, default='/data/home/yantao/zzx/NCPS/ncps/checkpoints/best_model_wc_small-world_64_20250421_144834.pt',
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--model_type', type=str, default='wc',
                        help='Type of model (wc)')
    parser.add_argument('--connection', type=str, default='small-world',
                        choices=['fc', 'ncp', 'small-world', 'hierarchical'],
                        help='Type of connection in the model')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size used in the model')
    parser.add_argument('--output_dir', type=str, default='./gradcam_results',
                        help='Directory to save the GradCAM visualizations')
    parser.add_argument('--use_ode', action='store_true',
                        help='Whether the model uses ODE solver (RK4)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step for Wilson-Cowan integration')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    parser.add_argument('--analyze_val', action='store_true',
                        help='Analyze validation samples instead of game frames')
    parser.add_argument('--num_val_samples', type=int, default=8,
                        help='Number of validation samples to analyze')
    parser.add_argument('--target_layer', type=str, default=None,
                        help='Target convolutional layer for GradCAM (default: auto-detect)')
    parser.add_argument('--list_layers', action='store_true',
                        help='List all available layers in the model')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定设备
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(
        args.model_path, 
        args.model_type, 
        args.connection, 
        args.hidden_size,
        use_ode=args.use_ode,
        dt=args.dt
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # 如果要列出所有层
    if args.list_layers:
        print("\n可用层:")
        layers = list_all_layers(model)
        for name, layer_type in layers:
            print(f"- {name} ({layer_type})")
        return
    
    # 初始化 GradCAM，只关注卷积层
    grad_cam = WilsonCowanGradCAM(model, target_layer=args.target_layer)
    
    if args.analyze_val:
        # 加载验证数据集
        val_dataset = AtariCloningDataset("breakout", split="val")
        print("Analyzing validation samples...")
        analyze_validation_samples(
            model, 
            grad_cam, 
            val_dataset, 
            device, 
            output_dir / "validation_samples",
            num_samples=args.num_val_samples
        )
    else:
        # 初始化环境
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        env = wrap_deepmind(env)
        print("Analyzing game frames...")
        analyze_game_frame(model, grad_cam, env, device, output_dir / "game_frames")
    
    print(f"GradCAM visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()