import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
import matplotlib.gridspec as gridspec
from pathlib import Path
import matplotlib.ticker as mtick
from datetime import datetime
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

# Set up global visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# Custom color palettes
color_palettes = {
    "models": {
        "ltc": "#1f77b4",  # blue
        "cfc": "#ff7f0e",  # orange  
        "lstm": "#2ca02c", # green
        "rnn": "#d62728",   # red
        "wc": "#7f7f7f",   # gray
    },
    "connections": {
        "fc": "#9467bd",    # purple
        "ncp": "#8c564b",   # brown
        "breakout": "#e377c2",  # pink
        "small-world": "#17becf",  # cyan
        "random": "#bcbd22",  # olive
        "hierarchical": "#7f7f7f",  # gray
    },
    "combined": sns.color_palette("tab10", 10),
    "sequential": sns.color_palette("viridis", 10),
    "paired": sns.color_palette("Paired", 10),
    "qualitative": sns.color_palette("Set3", 10)
}

# Create custom gradient colormaps
custom_cmap1 = LinearSegmentedColormap.from_list("blue_orange_red", 
                                               [(0, "#1f77b4"), (0.5, "#ffdd55"), (1, "#d62728")])
custom_cmap2 = LinearSegmentedColormap.from_list("green_purple", 
                                               [(0, "#2ca02c"), (0.5, "#c2c2c2"), (1, "#9467bd")])

def load_training_data(log_dir):
    """
    Load all training data from CSV logs
    
    Args:
        log_dir: Directory containing training logs
        
    Returns:
        Dictionary of dataframes with metrics data
    """
    print(f"Loading training data from {log_dir}...")
    
    # Find all metrics CSV files
    metric_files = glob.glob(os.path.join(log_dir, "*_metrics.csv"))
    batch_files = glob.glob(os.path.join(log_dir, "*_batch_metrics.csv"))
    info_files = glob.glob(os.path.join(log_dir, "*_info.json"))
    
    metrics_data = []
    batch_data = []
    model_info = []
    
    # Load metrics data
    for file in metric_files:
        try:
            df = pd.read_csv(file)
            
            # Extract model type, connection type, and hidden size from filename
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 4:
                model_type = parts[0]
                connection_type = parts[1]
                hidden_size = parts[2]
                
                # Add model info to DataFrame
                df['model_type'] = model_type
                df['connection_type'] = connection_type
                df['hidden_size'] = hidden_size
                df['run_id'] = '_'.join(parts[:-1])  # Everything except "_metrics.csv"
                
                metrics_data.append(df)
        except Exception as e:
            print(f"Error loading metrics file {file}: {e}")
    
    # Load batch metrics data
    for file in batch_files:
        try:
            df = pd.read_csv(file)
            
            # Extract model type, connection type, and hidden size from filename
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 4:
                model_type = parts[0]
                connection_type = parts[1]
                hidden_size = parts[2]
                
                # Add model info to DataFrame
                df['model_type'] = model_type
                df['connection_type'] = connection_type
                df['hidden_size'] = hidden_size
                df['run_id'] = '_'.join(parts[:-2])  # Everything except "_batch_metrics.csv"
                
                batch_data.append(df)
        except Exception as e:
            print(f"Error loading batch metrics file {file}: {e}")
    
    # Load model info data
    for file in info_files:
        try:
            with open(file, 'r') as f:
                info = json.load(f)
            
            # Extract run_id from filename
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 4:
                run_id = '_'.join(parts[:-1])  # Everything except "_info.json"
                info['run_id'] = run_id
                
                model_info.append(info)
        except Exception as e:
            print(f"Error loading info file {file}: {e}")
    
    # Combine all metrics dataframes
    metrics_df = pd.concat(metrics_data, ignore_index=True) if metrics_data else pd.DataFrame()
    batch_df = pd.concat(batch_data, ignore_index=True) if batch_data else pd.DataFrame()
    info_df = pd.DataFrame(model_info) if model_info else pd.DataFrame()
    
    print(f"Loaded data for {len(metrics_data)} models")
    
    return {
        "metrics": metrics_df,
        "batch": batch_df,
        "info": info_df
    }

def compute_evaluation_metrics(data):
    """
    Compute comprehensive evaluation metrics for all models
    
    Args:
        data: Dictionary with metrics, batch data and model info
        
    Returns:
        DataFrame with computed metrics
    """
    print("Computing evaluation metrics...")
    metrics_df = data["metrics"]
    info_df = data["info"]
    
    # Create a list to store the computed metrics
    model_metrics = []
    
    # Process each unique model run
    for run_id in metrics_df['run_id'].unique():
        run_data = metrics_df[metrics_df['run_id'] == run_id]
        
        # Skip if the run has no data
        if len(run_data) == 0:
            continue
        
        # Get model information
        model_type = run_data['model_type'].iloc[0]
        connection_type = run_data['connection_type'].iloc[0]
        hidden_size = run_data['hidden_size'].iloc[0]
        
        # Find best metrics
        if run_data['val_accuracy'].notna().any():  # Check if there are any non-NaN values
            best_val_acc_idx = run_data['val_accuracy'].idxmax()
            best_val_acc = run_data.loc[best_val_acc_idx, 'val_accuracy']
            best_val_acc_epoch = run_data.loc[best_val_acc_idx, 'epoch']
        else:
            best_val_acc = np.nan
            best_val_acc_epoch = np.nan
        
        # For mean return
        if run_data['mean_return'].notna().any():  # Check if there are any non-NaN values
            best_return_idx = run_data['mean_return'].idxmax()
            best_return = run_data.loc[best_return_idx, 'mean_return']
            best_return_epoch = run_data.loc[best_return_idx, 'epoch']
        else:
            best_return = np.nan
            best_return_epoch = np.nan
        
        # Compute convergence metrics
        # Speed of convergence: Epoch where validation accuracy reaches 95% of its maximum
        if np.isnan(best_val_acc):
            convergence_speed = np.nan
        else:
            convergence_threshold = 0.95 * best_val_acc
            converged_epochs = run_data[run_data['val_accuracy'] >= convergence_threshold]['epoch']
            convergence_speed = converged_epochs.min() if len(converged_epochs) > 0 else np.nan
        
        # Compute stability metrics
        # Stability: Standard deviation of validation accuracy in last 25% of epochs
        last_quarter_start = int(0.75 * len(run_data))
        if last_quarter_start < len(run_data):
            valid_values = run_data['val_accuracy'][last_quarter_start:].dropna()
            stability = valid_values.std() if len(valid_values) > 0 else np.nan
        else:
            stability = np.nan
        
        # Compute efficiency metrics
        # Time efficiency: Average time per epoch
        time_efficiency = run_data['epoch_time'].mean()
        
        # Compute gap between train and validation accuracy
        # Overfitting tendancy: Difference between training and validation accuracy
        overfitting = (run_data['train_accuracy'] - run_data['val_accuracy']).mean()
        
        # Compute final metrics (last epoch)
        final_epoch = run_data['epoch'].max()
        final_val_acc = run_data[run_data['epoch'] == final_epoch]['val_accuracy'].values[0]
        final_train_acc = run_data[run_data['epoch'] == final_epoch]['train_accuracy'].values[0]
        final_return = run_data[run_data['epoch'] == final_epoch]['mean_return'].values[0]
        
        # Compute improvement rate
        # How quickly the model improves: slope of the validation accuracy curve
        if len(run_data) >= 5:  # Need at least a few points for meaningful slope
            x = run_data['epoch'].values
            y = run_data['val_accuracy'].values
            slope, _, _, _, _ = stats.linregress(x, y)
            improvement_rate = slope
        else:
            improvement_rate = np.nan
        
        # Compute return stability
        return_stability = run_data['mean_return'].std()
        
        # Compute area under the accuracy curve (AUC)
        # Higher AUC indicates better overall performance across all epochs
        val_auc = np.trapz(run_data['val_accuracy'], run_data['epoch'])
        
        # Get parameter count from info if available
        trainable_params = None
        if not info_df.empty and 'run_id' in info_df.columns:
            info_row = info_df[info_df['run_id'] == run_id]
            if not info_row.empty and 'trainable_parameters' in info_row.columns:
                trainable_params = info_row['trainable_parameters'].iloc[0]
        
        # Compute metric per parameter (if parameter count is available)
        if trainable_params:
            efficiency = best_val_acc / np.log10(trainable_params) if trainable_params > 0 else np.nan
        else:
            efficiency = np.nan
        
        # Combine all metrics
        model_metric = {
            'run_id': run_id,
            'model_type': model_type,
            'connection_type': connection_type,
            'hidden_size': hidden_size,
            'best_val_acc': best_val_acc,
            'best_val_acc_epoch': best_val_acc_epoch,
            'best_return': best_return,
            'best_return_epoch': best_return_epoch,
            'convergence_speed': convergence_speed,
            'stability': stability,
            'time_efficiency': time_efficiency,
            'overfitting': overfitting,
            'final_val_acc': final_val_acc,
            'final_train_acc': final_train_acc,
            'final_return': final_return,
            'improvement_rate': improvement_rate,
            'return_stability': return_stability,
            'val_auc': val_auc,
            'parameter_efficiency': efficiency,
            'trainable_params': trainable_params
        }
        
        model_metrics.append(model_metric)
    
    # Create the final metrics DataFrame
    eval_metrics_df = pd.DataFrame(model_metrics)
    
    print(f"Computed metrics for {len(eval_metrics_df)} models")
    return eval_metrics_df

def create_model_comparison_visualizations(data, eval_metrics, output_dir):
    """
    Create comprehensive model comparison visualizations
    
    Args:
        data: Dictionary with metrics, batch data and model info
        eval_metrics: DataFrame with computed evaluation metrics
        output_dir: Directory to save visualizations
    """
    print("Creating model comparison visualizations...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics_df = data["metrics"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Learning curves by model type
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    for model_type in sorted(metrics_df['model_type'].unique()):
        model_data = metrics_df[metrics_df['model_type'] == model_type]
        sns.lineplot(data=model_data, x='epoch', y='val_accuracy', 
                   label=f"{model_type.upper()}", 
                   color=color_palettes["models"].get(model_type.lower(), "#333333"))
    
    plt.title('Validation Accuracy by Model Type', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.legend(title="Model Type", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for model_type in sorted(metrics_df['model_type'].unique()):
        model_data = metrics_df[metrics_df['model_type'] == model_type]
        sns.lineplot(data=model_data, x='epoch', y='mean_return', 
                   label=f"{model_type.upper()}", 
                   color=color_palettes["models"].get(model_type.lower(), "#333333"))
    
    plt.title('Mean Return by Model Type', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Return', fontsize=14)
    plt.legend(title="Model Type", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"learning_curves_by_model_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 2. Learning curves by connection type
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    for conn_type in sorted(metrics_df['connection_type'].unique()):
        conn_data = metrics_df[metrics_df['connection_type'] == conn_type]
        sns.lineplot(data=conn_data, x='epoch', y='val_accuracy', 
                   label=f"{conn_type.upper()}", 
                   color=color_palettes["connections"].get(conn_type.lower(), "#333333"))
    
    plt.title('Validation Accuracy by Connection Type', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.legend(title="Connection Type", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for conn_type in sorted(metrics_df['connection_type'].unique()):
        conn_data = metrics_df[metrics_df['connection_type'] == conn_type]
        sns.lineplot(data=conn_data, x='epoch', y='mean_return', 
                   label=f"{conn_type.upper()}", 
                   color=color_palettes["connections"].get(conn_type.lower(), "#333333"))
    
    plt.title('Mean Return by Connection Type', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Return', fontsize=14)
    plt.legend(title="Connection Type", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"learning_curves_by_connection_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 3. Bar charts of best validation accuracy and best return
    plt.figure(figsize=(16, 10))
    
    # Create a categorical variable for model and connection combination
    eval_metrics['model_connection'] = eval_metrics['model_type'] + '-' + eval_metrics['connection_type']
    
    # Sort by best validation accuracy
    sorted_data = eval_metrics.sort_values('best_val_acc', ascending=False)
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(sorted_data['model_connection'], sorted_data['best_val_acc'], 
                  color=[color_palettes['models'].get(m.lower(), "#333333") for m in sorted_data['model_type']])
    
    plt.title('Best Validation Accuracy by Model Configuration', fontsize=16)
    plt.xlabel('Model Configuration (Model-Connection)', fontsize=14)
    plt.ylabel('Best Validation Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Sort by best return
    sorted_data = eval_metrics.sort_values('best_return', ascending=False)
    
    plt.subplot(2, 1, 2)
    bars = plt.bar(sorted_data['model_connection'], sorted_data['best_return'], 
                  color=[color_palettes['connections'].get(c.lower(), "#333333") for c in sorted_data['connection_type']])
    
    plt.title('Best Mean Return by Model Configuration', fontsize=16)
    plt.xlabel('Model Configuration (Model-Connection)', fontsize=14)
    plt.ylabel('Best Mean Return', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"best_metrics_by_model_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 4. Scatterplot: Parameter Efficiency vs. Performance
    if not eval_metrics.empty and 'trainable_params' in eval_metrics.columns:
        plt.figure(figsize=(12, 10))
        
        valid_data = eval_metrics[eval_metrics['trainable_params'].notnull()]
        
        # Size based on hidden_size
        sizes = valid_data['hidden_size'].astype(float) * 3
        
        # Create scatter plot
        scatter = plt.scatter(valid_data['trainable_params'], valid_data['best_val_acc'],
                            s=sizes, 
                            c=[color_palettes['models'].get(m.lower(), "#333333") for m in valid_data['model_type']], 
                            alpha=0.7)
        
        # Add labels for each point
        for i, row in valid_data.iterrows():
            plt.annotate(f"{row['model_type']}-{row['connection_type']}",
                       (row['trainable_params'], row['best_val_acc']),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8)
        
        plt.xscale('log')
        plt.title('Model Efficiency: Performance vs. Parameter Count', fontsize=16)
        plt.xlabel('Number of Trainable Parameters (log scale)', fontsize=14)
        plt.ylabel('Best Validation Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Create a custom legend for model types
        model_types = valid_data['model_type'].unique()
        legend_elements = [Patch(facecolor=color_palettes['models'].get(m.lower(), "#333333"), 
                                edgecolor='black', alpha=0.7, label=m.upper()) 
                          for m in model_types]
        plt.legend(handles=legend_elements, title="Model Type", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parameter_efficiency_{timestamp}.png"), dpi=300)
        plt.close()
    
    # 5. Radar chart of model performance across different metrics
    if not eval_metrics.empty:
        # Select a subset of metrics for the radar chart
        radar_metrics = [
            'best_val_acc',
            'best_return',
            'convergence_speed',
            'stability',
            'time_efficiency',
            'parameter_efficiency'
        ]
        
        # Filter metrics that are available
        available_metrics = [m for m in radar_metrics if m in eval_metrics.columns]
        
        # Only proceed if we have enough metrics
        if len(available_metrics) >= 3:
            # Select model configurations to compare (choose top performers based on val_acc)
            top_models = eval_metrics.sort_values('best_val_acc', ascending=False).head(5)
            
            # For each metric, normalize to [0, 1] range
            normalized_data = top_models.copy()
            for metric in available_metrics:
                if normalized_data[metric].notnull().sum() > 1:  # At least 2 non-null values
                    if metric in ['time_efficiency']:  # Lower is better
                        max_val = normalized_data[metric].max()
                        min_val = normalized_data[metric].min()
                        if max_val != min_val:  # Avoid division by zero
                            normalized_data[metric] = 1 - ((normalized_data[metric] - min_val) / (max_val - min_val))
                    elif metric == 'convergence_speed':  # Lower is better
                        # Invert so that faster convergence (lower epoch number) gets a higher score
                        max_val = normalized_data[metric].max()
                        min_val = normalized_data[metric].min()
                        if max_val != min_val:  # Avoid division by zero
                            normalized_data[metric] = 1 - ((normalized_data[metric] - min_val) / (max_val - min_val))
                    else:  # Higher is better
                        max_val = normalized_data[metric].max()
                        min_val = normalized_data[metric].min()
                        if max_val != min_val:  # Avoid division by zero
                            normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
            
            # Create radar chart
            fig = plt.figure(figsize=(12, 10))
            
            # Create angle values for each metric
            N = len(available_metrics)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon
            
            # Create subplot with polar projection
            ax = fig.add_subplot(111, polar=True)
            
            # Plot each model configuration
            for i, (_, row) in enumerate(top_models.iterrows()):
                values = [row[metric] for metric in available_metrics]
                values += values[:1]  # Close the polygon
                
                model_name = f"{row['model_type']}-{row['connection_type']}"
                color = color_palettes['combined'][i % len(color_palettes['combined'])]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            # Set labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], fontsize=10)
            plt.title('Model Performance Comparison', fontsize=16)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"radar_chart_{timestamp}.png"), dpi=300)
            plt.close()
    
    # 6. Heatmap of model performance
    if not eval_metrics.empty:
        # Create a pivot table of model configurations and their performance
        if len(eval_metrics['model_type'].unique()) > 1 and len(eval_metrics['connection_type'].unique()) > 1:
            plt.figure(figsize=(12, 10))
            
            # Create heatmap for validation accuracy
            plt.subplot(2, 1, 1)
            pivot_acc = eval_metrics.pivot_table(values='best_val_acc', 
                                             index='model_type', 
                                             columns='connection_type',
                                             aggfunc='mean')
            
            sns.heatmap(pivot_acc, annot=True, cmap=custom_cmap1, fmt=".3f", linewidths=.5)
            plt.title('Best Validation Accuracy by Model Configuration', fontsize=16)
            plt.ylabel('Model Type', fontsize=14)
            plt.xlabel('Connection Type', fontsize=14)
            
            # Create heatmap for mean return
            plt.subplot(2, 1, 2)
            pivot_return = eval_metrics.pivot_table(values='best_return', 
                                                index='model_type', 
                                                columns='connection_type',
                                                aggfunc='mean')
            
            sns.heatmap(pivot_return, annot=True, cmap=custom_cmap2, fmt=".1f", linewidths=.5)
            plt.title('Best Mean Return by Model Configuration', fontsize=16)
            plt.ylabel('Model Type', fontsize=14)
            plt.xlabel('Connection Type', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"performance_heatmap_{timestamp}.png"), dpi=300)
            plt.close()
    
    # 7. Training time comparison
    plt.figure(figsize=(14, 7))
    
    # Group by model configuration and compute average training time
    time_data = eval_metrics.copy()
    time_data['model_connection'] = time_data['model_type'] + '-' + time_data['connection_type']
    time_data = time_data.sort_values('time_efficiency', ascending=True)  # Lower time is better
    
    # Create bar chart of average epoch time
    sns.barplot(
        x='model_connection', 
        y='time_efficiency', 
        data=time_data,
        palette=[color_palettes['models'].get(m.lower(), "#333333") for m in time_data['model_type']]
    )
    
    plt.title('Average Training Time per Epoch by Model Configuration', fontsize=16)
    plt.xlabel('Model Configuration (Model-Connection)', fontsize=14)
    plt.ylabel('Average Time per Epoch (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_time_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 8. Convergence speed comparison
    valid_convergence = eval_metrics[eval_metrics['convergence_speed'].notnull()]
    
    if not valid_convergence.empty:
        plt.figure(figsize=(14, 7))
        
        valid_convergence['model_connection'] = valid_convergence['model_type'] + '-' + valid_convergence['connection_type']
        valid_convergence = valid_convergence.sort_values('convergence_speed', ascending=True)  # Lower is better
        
        # Create bar chart of convergence speed
        sns.barplot(
            x='model_connection', 
            y='convergence_speed', 
            data=valid_convergence,
            palette=[color_palettes['models'].get(m.lower(), "#333333") for m in valid_convergence['model_type']]
        )
        
        plt.title('Convergence Speed by Model Configuration (Lower is Better)', fontsize=16)
        plt.xlabel('Model Configuration (Model-Connection)', fontsize=14)
        plt.ylabel('Convergence Epoch', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_speed_comparison_{timestamp}.png"), dpi=300)
        plt.close()
    
    # 9. Multi-metric comparison chart
    plt.figure(figsize=(18, 10))
    
    # Create a grid of small charts, each showing a different metric
    metrics_to_show = [
        ('best_val_acc', 'Best Validation Accuracy'),
        ('best_return', 'Best Mean Return'),
        ('convergence_speed', 'Convergence Speed (Lower is Better)'),
        ('stability', 'Stability (Lower Variation is Better)'),
        ('time_efficiency', 'Time Efficiency (Lower is Better)'),
        ('overfitting', 'Overfitting Tendency (Lower is Better)'),
        ('improvement_rate', 'Improvement Rate (Higher is Better)'),
        ('parameter_efficiency', 'Parameter Efficiency (Higher is Better)')
    ]
    
    # Create a layout with 2 rows and 4 columns
    gs = gridspec.GridSpec(2, 4)
    
    # Loop through metrics and create small charts
    for i, (metric, title) in enumerate(metrics_to_show):
        if metric in eval_metrics.columns and eval_metrics[metric].notnull().any():
            row, col = divmod(i, 4)
            ax = plt.subplot(gs[row, col])
            
            valid_data = eval_metrics[eval_metrics[metric].notnull()].copy()
            valid_data['model_connection'] = valid_data['model_type'] + '-' + valid_data['connection_type']
            
            # Determine sort order based on metric
            ascending = metric in ['convergence_speed', 'stability', 'time_efficiency', 'overfitting']
            valid_data = valid_data.sort_values(metric, ascending=ascending)
            
            # Create bar chart
            color_values = [color_palettes['qualitative'][i % len(color_palettes['qualitative'])] for i in range(len(valid_data))]
            bars = plt.bar(valid_data['model_connection'], valid_data[metric], color=color_values)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                adjustment = 0.05 * max(valid_data[metric]) if height > 0 else -0.1 * max(valid_data[metric])
                plt.text(bar.get_x() + bar.get_width()/2., height + adjustment,
                        f'{height:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            plt.title(title, fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tick_params(axis='y', labelsize=8)
            plt.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Multi-Metric Performance Comparison', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, f"multi_metric_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 10. Learning dynamics visualization
    # Select top models based on best validation accuracy
    top_models = eval_metrics.sort_values('best_val_acc', ascending=False).head(3)['run_id'].tolist()
    top_model_data = metrics_df[metrics_df['run_id'].isin(top_models)]
    
    if not top_model_data.empty:
        plt.figure(figsize=(16, 12))
        
        # Plot training vs validation accuracy
        plt.subplot(2, 2, 1)
        for run_id in top_models:
            model_data = metrics_df[metrics_df['run_id'] == run_id]
            model_info = f"{model_data['model_type'].iloc[0]}-{model_data['connection_type'].iloc[0]}"
            
            # Plot training accuracy
            sns.lineplot(data=model_data, x='epoch', y='train_accuracy', 
                       label=f"{model_info} (Train)", 
                       linestyle='-')
            
            # Plot validation accuracy
            sns.lineplot(data=model_data, x='epoch', y='val_accuracy', 
                       label=f"{model_info} (Val)", 
                       linestyle='--')
        
        plt.title('Training vs. Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Plot training vs validation loss
        plt.subplot(2, 2, 2)
        for run_id in top_models:
            model_data = metrics_df[metrics_df['run_id'] == run_id]
            model_info = f"{model_data['model_type'].iloc[0]}-{model_data['connection_type'].iloc[0]}"
            
            # Plot training loss
            sns.lineplot(data=model_data, x='epoch', y='train_loss', 
                       label=f"{model_info} (Train)", 
                       linestyle='-')
            
            # Plot validation loss
            sns.lineplot(data=model_data, x='epoch', y='val_loss', 
                       label=f"{model_info} (Val)", 
                       linestyle='--')
        
        plt.title('Training vs. Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Plot mean return
        plt.subplot(2, 2, 3)
        for run_id in top_models:
            model_data = metrics_df[metrics_df['run_id'] == run_id]
            model_info = f"{model_data['model_type'].iloc[0]}-{model_data['connection_type'].iloc[0]}"
            
            sns.lineplot(data=model_data, x='epoch', y='mean_return', 
                       label=model_info)
        
        plt.title('Mean Return Over Time', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Return', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Plot gap between train and validation accuracy (overfitting indicator)
        plt.subplot(2, 2, 4)
        for run_id in top_models:
            model_data = metrics_df[metrics_df['run_id'] == run_id]
            model_info = f"{model_data['model_type'].iloc[0]}-{model_data['connection_type'].iloc[0]}"
            
            # Calculate gap
            model_data['accuracy_gap'] = model_data['train_accuracy'] - model_data['val_accuracy']
            
            sns.lineplot(data=model_data, x='epoch', y='accuracy_gap', 
                       label=model_info)
        
        plt.title('Overfitting Indicator: Train-Val Accuracy Gap', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy Gap', fontsize=12)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"learning_dynamics_{timestamp}.png"), dpi=300)
        plt.close()
    
    print(f"Saved all model comparison visualizations to {output_dir}")
    
def create_detailed_model_visualizations(data, eval_metrics, output_dir, run_id=None):
    """
    Create detailed visualizations for a specific model run or all runs
    
    Args:
        data: Dictionary with metrics, batch data and model info
        eval_metrics: DataFrame with computed evaluation metrics
        output_dir: Directory to save visualizations
        run_id: Specific run ID to visualize (None for all runs)
    """
    metrics_df = data["metrics"]
    batch_df = data["batch"]
    
    # If run_id is specified, filter the data
    if run_id is not None:
        metrics_df = metrics_df[metrics_df['run_id'] == run_id]
        batch_df = batch_df[batch_df['run_id'] == run_id]
        
        if metrics_df.empty:
            print(f"No data found for run ID: {run_id}")
            return
    
    # Create directory for detailed visualizations
    detailed_dir = os.path.join(output_dir, "detailed_visualizations")
    Path(detailed_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each run separately
    for current_run_id in metrics_df['run_id'].unique():
        print(f"Creating detailed visualizations for run: {current_run_id}")
        
        run_metrics = metrics_df[metrics_df['run_id'] == current_run_id]
        run_batches = batch_df[batch_df['run_id'] == current_run_id] if not batch_df.empty else pd.DataFrame()
        
        # Create a subdirectory for this run
        run_dir = os.path.join(detailed_dir, current_run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        
        model_type = run_metrics['model_type'].iloc[0]
        connection_type = run_metrics['connection_type'].iloc[0]
        hidden_size = run_metrics['hidden_size'].iloc[0]
        
        # 1. Create comprehensive learning curves
        plt.figure(figsize=(16, 12))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        sns.lineplot(data=run_metrics, x='epoch', y='train_accuracy', 
                   label='Training Accuracy', color='blue', marker='o')
        sns.lineplot(data=run_metrics, x='epoch', y='val_accuracy', 
                   label='Validation Accuracy', color='green', marker='s')
        
        plt.title(f'Accuracy Over Time - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Plot loss
        plt.subplot(2, 2, 2)
        sns.lineplot(data=run_metrics, x='epoch', y='train_loss', 
                   label='Training Loss', color='red', marker='o')
        sns.lineplot(data=run_metrics, x='epoch', y='val_loss', 
                   label='Validation Loss', color='purple', marker='s')
        
        plt.title(f'Loss Over Time - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Plot mean return
        plt.subplot(2, 2, 3)
        sns.lineplot(data=run_metrics, x='epoch', y='mean_return', 
                   color='orange', marker='o')
        
        plt.title(f'Mean Return Over Time - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Return', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot epoch time
        plt.subplot(2, 2, 4)
        sns.lineplot(data=run_metrics, x='epoch', y='epoch_time', 
                   color='brown', marker='o')
        
        plt.title(f'Training Time per Epoch - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance: {model_type.upper()}-{connection_type.upper()} (Hidden Size: {hidden_size})', 
                  fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"learning_curves_{current_run_id}.png"), dpi=300)
        plt.close()
        
        # 2. Batch-level analysis (if batch data is available)
        if not run_batches.empty:
            plt.figure(figsize=(16, 8))
            
            # Plot batch loss
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=run_batches, x='global_step', y='batch_loss', 
                         alpha=0.3, color='blue', s=10)
            
            # Add smoothed line
            window_size = min(100, len(run_batches) // 10) if len(run_batches) > 0 else 1
            if len(run_batches) > window_size:
                run_batches['smooth_loss'] = run_batches['batch_loss'].rolling(window=window_size).mean()
                sns.lineplot(data=run_batches, x='global_step', y='smooth_loss', 
                           linewidth=2.5, color='red', 
                           label=f'Moving Average (window={window_size})')
            
            plt.title(f'Batch Loss During Training - {model_type}-{connection_type}', fontsize=14)
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            # Plot batch accuracy
            plt.subplot(1, 2, 2)
            sns.scatterplot(data=run_batches, x='global_step', y='batch_accuracy', 
                         alpha=0.3, color='green', s=10)
            
            # Add smoothed line
            if len(run_batches) > window_size:
                run_batches['smooth_acc'] = run_batches['batch_accuracy'].rolling(window=window_size).mean()
                sns.lineplot(data=run_batches, x='global_step', y='smooth_acc', 
                           linewidth=2.5, color='purple', 
                           label=f'Moving Average (window={window_size})')
            
            plt.title(f'Batch Accuracy During Training - {model_type}-{connection_type}', fontsize=14)
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"batch_analysis_{current_run_id}.png"), dpi=300)
            plt.close()
        
        # 3. Create heatmap showing the change in metrics over epochs
        metrics_to_include = ['train_accuracy', 'val_accuracy', 'train_loss', 'val_loss', 'mean_return', 'epoch_time']
        metrics_available = [m for m in metrics_to_include if m in run_metrics.columns]
        
        if len(metrics_available) > 0:
            # Scale each metric to [0, 1] range for better visualization
            normalized_metrics = run_metrics.copy()
            
            for metric in metrics_available:
                if normalized_metrics[metric].max() != normalized_metrics[metric].min():
                    if metric.endswith('loss') or metric == 'epoch_time':  # Lower is better
                        normalized_metrics[metric] = 1 - ((normalized_metrics[metric] - normalized_metrics[metric].min()) / 
                                                     (normalized_metrics[metric].max() - normalized_metrics[metric].min()))
                    else:  # Higher is better
                        normalized_metrics[metric] = ((normalized_metrics[metric] - normalized_metrics[metric].min()) / 
                                                  (normalized_metrics[metric].max() - normalized_metrics[metric].min()))
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            
            # Select only the metrics columns and transpose for better visualization
            heatmap_data = normalized_metrics[metrics_available].transpose()
            
            # Create the heatmap
            sns.heatmap(heatmap_data, cmap="viridis", annot=False, fmt=".2f", 
                      xticklabels=normalized_metrics['epoch'].values)
            
            plt.title(f'Metric Evolution Over Epochs - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Metric', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"metrics_heatmap_{current_run_id}.png"), dpi=300)
            plt.close()
            
        # 4. Create accuracy vs return scatter plot
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(run_metrics['val_accuracy'], run_metrics['mean_return'], 
                            c=run_metrics['epoch'], cmap='viridis', 
                            s=80, alpha=0.7)
        
        # Add labels for some points
        for i, row in run_metrics.iloc[::max(1, len(run_metrics)//10)].iterrows():  # Label every ~10% of points
            plt.annotate(f"E{int(row['epoch'])}", 
                       (row['val_accuracy'], row['mean_return']),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Epoch', fontsize=12)
        
        plt.title(f'Validation Accuracy vs. Mean Return - {model_type}-{connection_type} (Size: {hidden_size})', fontsize=16)
        plt.xlabel('Validation Accuracy', fontsize=14)
        plt.ylabel('Mean Return', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"accuracy_vs_return_{current_run_id}.png"), dpi=300)
        plt.close()
        
        # Create a summary file
        with open(os.path.join(run_dir, f"model_summary_{current_run_id}.txt"), 'w') as f:
            f.write(f"Model Summary: {model_type.upper()}-{connection_type.upper()} (Hidden Size: {hidden_size})\n")
            f.write("=" * 80 + "\n\n")
            
            # Get model performance from eval_metrics
            if not eval_metrics.empty:
                model_eval = eval_metrics[eval_metrics['run_id'] == current_run_id]
                
                if not model_eval.empty:
                    f.write("Performance Metrics:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Best Validation Accuracy: {model_eval['best_val_acc'].iloc[0]:.4f} (Epoch {int(model_eval['best_val_acc_epoch'].iloc[0])})\n")
                    f.write(f"Best Mean Return: {model_eval['best_return'].iloc[0]:.2f} (Epoch {int(model_eval['best_return_epoch'].iloc[0])})\n")
                    f.write(f"Convergence Speed: {model_eval['convergence_speed'].iloc[0]:.2f} epochs\n")
                    f.write(f"Stability (Std Dev): {model_eval['stability'].iloc[0]:.4f}\n")
                    f.write(f"Average Time per Epoch: {model_eval['time_efficiency'].iloc[0]:.2f} seconds\n")
                    f.write(f"Overfitting Tendency: {model_eval['overfitting'].iloc[0]:.4f}\n")
                    f.write(f"Final Validation Accuracy: {model_eval['final_val_acc'].iloc[0]:.4f}\n")
                    f.write(f"Final Mean Return: {model_eval['final_return'].iloc[0]:.2f}\n\n")
            
            f.write("Training Progress:\n")
            f.write("-" * 40 + "\n")
            f.write("Epoch  Train_Acc  Val_Acc  Train_Loss  Val_Loss  Mean_Return  Time\n")
            
            for _, row in run_metrics.iterrows():
                f.write(f"{int(row['epoch']):5d}  {row['train_accuracy']:.4f}   {row['val_accuracy']:.4f}   "
                      f"{row['train_loss']:.4f}    {row['val_loss']:.4f}   {row['mean_return']:.2f}       "
                      f"{row['epoch_time']:.2f}s\n")

def create_summary_report(data, eval_metrics, output_dir):
    """
    Create a comprehensive summary report of all models
    
    Args:
        data: Dictionary with metrics, batch data and model info
        eval_metrics: DataFrame with computed evaluation metrics
        output_dir: Directory to save the report
    """
    print("Creating summary report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"model_evaluation_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("Atari Game Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview of models evaluated
        f.write("Models Evaluated:\n")
        f.write("-" * 40 + "\n")
        
        if not eval_metrics.empty:
            model_counts = eval_metrics.groupby(['model_type', 'connection_type']).size().reset_index(name='count')
            for _, row in model_counts.iterrows():
                f.write(f"- {row['model_type'].upper()}-{row['connection_type'].upper()}: {row['count']} run(s)\n")
        
        f.write("\n")
        
        # Best models section
        f.write("Best Performing Models:\n")
        f.write("-" * 40 + "\n")
        
        if not eval_metrics.empty:
            # Best validation accuracy
            best_val_acc = eval_metrics.loc[eval_metrics['best_val_acc'].idxmax()]
            f.write(f"Best Validation Accuracy: {best_val_acc['best_val_acc']:.4f}\n")
            f.write(f"  Model: {best_val_acc['model_type'].upper()}-{best_val_acc['connection_type'].upper()} "
                  f"(Hidden Size: {best_val_acc['hidden_size']})\n")
            f.write(f"  Run ID: {best_val_acc['run_id']}\n\n")
            
            # Best mean return
            best_return = eval_metrics.loc[eval_metrics['best_return'].idxmax()]
            f.write(f"Best Mean Return: {best_return['best_return']:.2f}\n")
            f.write(f"  Model: {best_return['model_type'].upper()}-{best_return['connection_type'].upper()} "
                  f"(Hidden Size: {best_return['hidden_size']})\n")
            f.write(f"  Run ID: {best_return['run_id']}\n\n")
            
            # Fastest convergence
            if 'convergence_speed' in eval_metrics.columns:
                valid_conv = eval_metrics[eval_metrics['convergence_speed'].notnull()]
                if not valid_conv.empty:
                    fastest_conv = valid_conv.loc[valid_conv['convergence_speed'].idxmin()]
                    f.write(f"Fastest Convergence: {fastest_conv['convergence_speed']:.2f} epochs\n")
                    f.write(f"  Model: {fastest_conv['model_type'].upper()}-{fastest_conv['connection_type'].upper()} "
                          f"(Hidden Size: {fastest_conv['hidden_size']})\n")
                    f.write(f"  Run ID: {fastest_conv['run_id']}\n\n")
            
            # Most efficient (time)
            fastest_training = eval_metrics.loc[eval_metrics['time_efficiency'].idxmin()]
            f.write(f"Most Time-Efficient: {fastest_training['time_efficiency']:.2f} seconds per epoch\n")
            f.write(f"  Model: {fastest_training['model_type'].upper()}-{fastest_training['connection_type'].upper()} "
                  f"(Hidden Size: {fastest_training['hidden_size']})\n")
            f.write(f"  Run ID: {fastest_training['run_id']}\n\n")
            
            # Best parameter efficiency
            if 'parameter_efficiency' in eval_metrics.columns:
                valid_eff = eval_metrics[eval_metrics['parameter_efficiency'].notnull()]
                if not valid_eff.empty:
                    best_efficiency = valid_eff.loc[valid_eff['parameter_efficiency'].idxmax()]
                    f.write(f"Best Parameter Efficiency: {best_efficiency['parameter_efficiency']:.4f}\n")
                    f.write(f"  Model: {best_efficiency['model_type'].upper()}-{best_efficiency['connection_type'].upper()} "
                          f"(Hidden Size: {best_efficiency['hidden_size']})\n")
                    f.write(f"  Run ID: {best_efficiency['run_id']}\n\n")
        
        # Comparative analysis
        f.write("Comparative Analysis:\n")
        f.write("-" * 40 + "\n")
        
        if not eval_metrics.empty:
            # Compare model types
            f.write("Performance by Model Type (Average):\n")
            model_type_stats = eval_metrics.groupby('model_type').agg({
                'best_val_acc': 'mean',
                'best_return': 'mean',
                'time_efficiency': 'mean'
            }).reset_index()
            
            for _, row in model_type_stats.iterrows():
                f.write(f"  {row['model_type'].upper()}: Val Acc={row['best_val_acc']:.4f}, "
                      f"Return={row['best_return']:.2f}, Time={row['time_efficiency']:.2f}s\n")
            
            f.write("\n")
            
            # Compare connection types
            f.write("Performance by Connection Type (Average):\n")
            conn_type_stats = eval_metrics.groupby('connection_type').agg({
                'best_val_acc': 'mean',
                'best_return': 'mean',
                'time_efficiency': 'mean'
            }).reset_index()
            
            for _, row in conn_type_stats.iterrows():
                f.write(f"  {row['connection_type'].upper()}: Val Acc={row['best_val_acc']:.4f}, "
                      f"Return={row['best_return']:.2f}, Time={row['time_efficiency']:.2f}s\n")
            
            f.write("\n")
            
            # Top 5 configurations based on validation accuracy
            f.write("Top 5 Configurations by Validation Accuracy:\n")
            top5_acc = eval_metrics.sort_values('best_val_acc', ascending=False).head(5)
            for i, (_, row) in enumerate(top5_acc.iterrows(), 1):
                f.write(f"  {i}. {row['model_type'].upper()}-{row['connection_type'].upper()} "
                      f"(Size: {row['hidden_size']}): {row['best_val_acc']:.4f}\n")
            
            f.write("\n")
            
            # Top 5 configurations based on mean return
            f.write("Top 5 Configurations by Mean Return:\n")
            top5_return = eval_metrics.sort_values('best_return', ascending=False).head(5)
            for i, (_, row) in enumerate(top5_return.iterrows(), 1):
                f.write(f"  {i}. {row['model_type'].upper()}-{row['connection_type'].upper()} "
                      f"(Size: {row['hidden_size']}): {row['best_return']:.2f}\n")
        
        # Conclusion and recommendations
        f.write("\n")
        f.write("Conclusions and Recommendations:\n")
        f.write("-" * 40 + "\n")
        
        if not eval_metrics.empty:
            # Best overall model
            # Calculate a combined score: 0.7 * normalized_val_acc + 0.3 * normalized_return
            if len(eval_metrics) > 1:  # Need at least 2 models to normalize
                eval_copy = eval_metrics.copy()
                
                # Normalize validation accuracy
                acc_min = eval_copy['best_val_acc'].min()
                acc_max = eval_copy['best_val_acc'].max()
                if acc_max > acc_min:
                    eval_copy['norm_val_acc'] = (eval_copy['best_val_acc'] - acc_min) / (acc_max - acc_min)
                else:
                    eval_copy['norm_val_acc'] = 1.0
                
                # Normalize mean return
                ret_min = eval_copy['best_return'].min()
                ret_max = eval_copy['best_return'].max()
                if ret_max > ret_min:
                    eval_copy['norm_return'] = (eval_copy['best_return'] - ret_min) / (ret_max - ret_min)
                else:
                    eval_copy['norm_return'] = 1.0
                
                # Calculate combined score
                eval_copy['combined_score'] = 0.7 * eval_copy['norm_val_acc'] + 0.3 * eval_copy['norm_return']
                
                # Get best overall model
                best_overall = eval_copy.loc[eval_copy['combined_score'].idxmax()]
                
                f.write(f"Best Overall Model: {best_overall['model_type'].upper()}-{best_overall['connection_type'].upper()} "
                      f"(Hidden Size: {best_overall['hidden_size']})\n")
                f.write(f"  Validation Accuracy: {best_overall['best_val_acc']:.4f}\n")
                f.write(f"  Mean Return: {best_overall['best_return']:.2f}\n")
                f.write(f"  Combined Score: {best_overall['combined_score']:.4f}\n\n")
            
            # General findings
            f.write("General Findings:\n")
            
            # Find best model type
            best_model_type = model_type_stats.loc[model_type_stats['best_val_acc'].idxmax(), 'model_type']
            f.write(f"- {best_model_type.upper()} models tend to perform best on average.\n")
            
            # Find best connection type
            best_conn_type = conn_type_stats.loc[conn_type_stats['best_val_acc'].idxmax(), 'connection_type']
            f.write(f"- {best_conn_type.upper()} connection type shows superior performance.\n")
            
            # Check if larger hidden sizes generally perform better
            if 'hidden_size' in eval_metrics.columns and eval_metrics['hidden_size'].nunique() > 1:
                size_corr = eval_metrics[['hidden_size', 'best_val_acc']].corr().iloc[0, 1]
                if size_corr > 0.3:
                    f.write("- Larger hidden sizes tend to yield better performance.\n")
                elif size_corr < -0.3:
                    f.write("- Smaller hidden sizes tend to yield better performance.\n")
                else:
                    f.write("- Hidden size does not strongly correlate with performance.\n")
            
            # Check relationship between validation accuracy and mean return
            acc_return_corr = eval_metrics[['best_val_acc', 'best_return']].corr().iloc[0, 1]
            if abs(acc_return_corr) > 0.5:
                if acc_return_corr > 0:
                    f.write("- Models with higher validation accuracy tend to achieve better mean returns.\n")
                else:
                    f.write("- Higher validation accuracy does not necessarily translate to better game performance.\n")
            else:
                f.write("- Validation accuracy and mean return appear to be weakly correlated.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("End of Report")
    
    print(f"Summary report saved to: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate Atari game model training results')
    parser.add_argument('--log_dir', type=str, default='./wclogs_adjust',
                       help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='./Atari_evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Generate detailed analysis for each model')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Specific run ID to analyze (optional)')
    parser.add_argument('--create_report', action='store_true', default=True,
                       help='Generate summary report')
    parser.add_argument('--visualize_best_only', action='store_true',
                       help='Only visualize the best performing models')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ATARI MODEL EVALUATION TOOL")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all training data
    data = load_training_data(args.log_dir)
    
    if data["metrics"].empty:
        print("No training data found in the specified directory.")
        return
    
    print(f"\nFound data for {data['metrics']['run_id'].nunique()} training runs")
    print(f"Model types: {', '.join(sorted(data['metrics']['model_type'].unique()))}")
    print(f"Connection types: {', '.join(sorted(data['metrics']['connection_type'].unique()))}")
    print(f"Hidden sizes: {', '.join(sorted(data['metrics']['hidden_size'].astype(str).unique()))}")
    
    # Compute evaluation metrics
    eval_metrics = compute_evaluation_metrics(data)
    
    # If visualize_best_only is enabled, filter to keep only the top performers
    if args.visualize_best_only and not eval_metrics.empty:
        # Keep top 3 by validation accuracy and top 3 by mean return
        top_by_acc = eval_metrics.nlargest(3, 'best_val_acc')['run_id'].tolist()
        top_by_return = eval_metrics.nlargest(3, 'best_return')['run_id'].tolist()
        
        # Combine the lists (removing duplicates)
        top_runs = list(set(top_by_acc + top_by_return))
        
        # Filter the metrics DataFrame
        eval_metrics = eval_metrics[eval_metrics['run_id'].isin(top_runs)]
        
        # Filter the metrics data in the data dictionary
        data["metrics"] = data["metrics"][data["metrics"]['run_id'].isin(top_runs)]
        
        if not data["batch"].empty:
            data["batch"] = data["batch"][data["batch"]['run_id'].isin(top_runs)]
            
        print(f"\nFiltering to show only the top {len(top_runs)} performing models")
    
    # Create visualizations directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    print(f"\nGenerating model comparison visualizations in {vis_output_dir}...")
    
    # Create model comparison visualizations
    create_model_comparison_visualizations(data, eval_metrics, vis_output_dir)
    
    # Create detailed visualizations if requested
    if args.detailed_analysis:
        print("\nGenerating detailed model visualizations...")
        create_detailed_model_visualizations(data, eval_metrics, vis_output_dir, args.run_id)
    
    # Create summary report if requested
    if args.create_report:
        print("\nGenerating summary report...")
        report_path = create_summary_report(data, eval_metrics, vis_output_dir)
        print(f"Summary report saved to: {report_path}")
    
    # Save the evaluation metrics DataFrame for future reference
    metrics_output_path = os.path.join(vis_output_dir, f"evaluation_metrics_{timestamp}.csv")
    eval_metrics.to_csv(metrics_output_path, index=False)
    print(f"\nEvaluation metrics saved to: {metrics_output_path}")
    
    # Create a README file with instructions
    readme_path = os.path.join(vis_output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Atari Game Model Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Contents:\n")
        f.write("- Model comparison visualizations\n")
        if args.detailed_analysis:
            f.write("- Detailed model visualizations in the 'detailed_visualizations' directory\n")
        if args.create_report:
            f.write("- Summary report\n")
        f.write("- Evaluation metrics CSV file\n\n")
        f.write("To generate new visualizations with different options, run:\n")
        f.write("python AtariEvaluate.py --log_dir LOG_DIR [--output_dir OUTPUT_DIR] [--detailed_analysis] [--run_id RUN_ID] [--visualize_best_only]\n")
    
    print("\n" + "="*80)
    print(f"Evaluation complete! Results saved to: {vis_output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()