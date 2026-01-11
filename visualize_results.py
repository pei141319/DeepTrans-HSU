import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

def plot_training_metrics(dataset_name='samson'):
    """
    绘制训练过程中的SAD和RMSE指标
    
    Args:
        dataset_name: 数据集名称 ('samson', 'jasper', 'dc', etc.)
    """
    # 加载损失数据
    loss_file = f"trans_mod_{dataset_name}/{dataset_name}_losses.mat"
    
    if os.path.exists(loss_file):
        losses_data = sio.loadmat(loss_file)
        losses = losses_data['losses'][0]
        
        # 创建epoch数组
        epochs = range(len(losses))
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制总损失
        plt.subplot(2, 2, 1)
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Total Loss')
        plt.title(f'Training Loss - {dataset_name.upper()} Dataset')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 如果有单独的re_loss和sad_loss数据，也可以绘制
        # 但由于我们当前的实现中这些是在训练过程中计算的，
        # 而不是存储在.mat文件中，所以我们只能绘制总损失
        
    else:
        print(f"Loss file {loss_file} not found.")
    
    # 创建第二个图，展示SAD和RMSE指标（如果有的话）
    plt.subplot(2, 2, 2)
    # 使用之前训练输出的指标
    # 由于SAD和RMSE是在训练完成后一次性计算的，我们需要模拟这些指标随训练进程的变化
    epochs_for_metrics = range(0, 200, 10)  # 每10个epoch记录一次
    # 模拟SAD值随训练减少的趋势
    simulated_sad = [0.9 - 0.002*i for i in epochs_for_metrics]  # 模拟递减趋势
    # 模拟RMSE值随训练减少的趋势
    simulated_rmse = [0.45 - 0.0015*i for i in epochs_for_metrics]  # 模拟递减趋势
    
    plt.plot(epochs_for_metrics, simulated_sad, 'r-', linewidth=2, label='SAD')
    plt.plot(epochs_for_metrics, simulated_rmse, 'g-', linewidth=2, label='RMSE')
    plt.title(f'Simulated Metrics Evolution - {dataset_name.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 展示最终结果的柱状图
    plt.subplot(2, 2, 3)
    metrics = ['SAD', 'RMSE']
    values = [0.717, 0.379]  # 从你的训练结果中获取的值
    colors = ['red', 'green']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title(f'Final Metrics - {dataset_name.upper()}')
    plt.ylabel('Value')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 创建类别的SAD和RMSE比较图
    plt.subplot(2, 2, 4)
    classes = ['Class 1', 'Class 2', 'Class 3']
    sad_values = [0.550, 0.907, 0.692]  # 从你的训练结果中获取
    rmse_values = [0.379, 0.389, 0.370]  # 从你的训练结果中获取
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, sad_values, width, label='SAD', alpha=0.7)
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Value')
    plt.title(f'Per-Class Metrics - {dataset_name.upper()}')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'trans_mod_{dataset_name}/{dataset_name}_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_metrics(dataset_name='samson'):
    """
    绘制详细的SAD和RMSE指标对比
    """
    # 从训练结果中提取的数据
    classes = ['Class 1', 'Class 2', 'Class 3']
    sad_values = [0.550, 0.907, 0.692]  # 从你的训练结果中获取
    rmse_values = [0.379, 0.389, 0.370]  # 从你的训练结果中获取
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SAD值对比
    bars1 = ax1.bar(classes, sad_values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax1.set_title(f'Spectral Angle Distance (SAD) - {dataset_name.upper()}', fontsize=14)
    ax1.set_ylabel('SAD Value (radians)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars1, sad_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE值对比
    bars2 = ax2.bar(classes, rmse_values, color=['gold', 'orange', 'yellow'], alpha=0.7)
    ax2.set_title(f'Root Mean Square Error (RMSE) - {dataset_name.upper()}', fontsize=14)
    ax2.set_ylabel('RMSE Value')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'trans_mod_{dataset_name}/{dataset_name}_detailed_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_samson_results():
    """
    可视化Samson数据集的训练结果
    """
    # 从你的训练输出中提取的数据
    dataset_name = "samson"
    
    # 最终指标
    final_sad_mean = 0.717  # 弧度
    final_rmse_mean = 0.379
    
    # 各类别的指标
    class_sad_values = [0.550, 0.907, 0.692]  # Class 1, 2, 3
    class_rmse_values = [0.379, 0.389, 0.370]  # Class 1, 2, 3
    
    classes = ['Class 1', 'Class 2', 'Class 3']
    
    # 创建综合可视化图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各类别SAD对比
    bars1 = ax1.bar(classes, class_sad_values, color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.7)
    ax1.set_title(f'Spectral Angle Distance (SAD) per Class - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('SAD Value (radians)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, class_sad_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. 各类别RMSE对比
    bars2 = ax2.bar(classes, class_rmse_values, color=['#FFCC99', '#FF99CC', '#CC99FF'], alpha=0.7)
    ax2.set_title(f'Root Mean Square Error (RMSE) per Class - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE Value')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, class_rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. 综合指标对比
    metrics = ['Mean SAD', 'Mean RMSE']
    metric_values = [final_sad_mean, final_rmse_mean]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars3 = ax3.bar(metrics, metric_values, color=colors, alpha=0.7)
    ax3.set_title(f'Mean Metrics Comparison - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars3, metric_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 4. 指标雷达图比较
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    # 完成闭环
    class_sad_values_plot = class_sad_values + [class_sad_values[0]]
    class_rmse_values_plot = class_rmse_values + [class_rmse_values[0]]
    angles_plot = angles + [angles[0]]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles_plot, class_sad_values_plot, 'o-', linewidth=2, label='SAD', color='red')
    ax4.fill(angles_plot, class_sad_values_plot, alpha=0.25, color='red')
    ax4.plot(angles_plot, class_rmse_values_plot, 'o-', linewidth=2, label='RMSE', color='blue')
    ax4.fill(angles_plot, class_rmse_values_plot, alpha=0.25, color='blue')
    
    ax4.set_xticks(angles)
    ax4.set_xticklabels(classes)
    ax4.set_title(f'Radar Chart: SAD vs RMSE per Class - {dataset_name.upper()}', 
                  fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'trans_mod_{dataset_name}/{dataset_name}_comprehensive_metrics.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print(f"\n=== {dataset_name.upper()} DATASET RESULTS SUMMARY ===")
    print(f"Mean SAD: {final_sad_mean:.3f} radians ({np.degrees(final_sad_mean):.2f}°)")
    print(f"Mean RMSE: {final_rmse_mean:.3f}")
    print("\nPer-class metrics:")
    for i, (sad_val, rmse_val) in enumerate(zip(class_sad_values, class_rmse_values)):
        print(f"  Class {i+1}: SAD = {sad_val:.3f}, RMSE = {rmse_val:.3f}")

def interpret_metrics():
    """
    解释指标含义
    """
    print("\n=== METRICS INTERPRETATION ===")
    print("Spectral Angle Distance (SAD):")
    print("  • Measures spectral similarity between estimated and true endmembers")
    print("  • Lower values indicate better spectral preservation")
    print("  • Values close to 0 are ideal, typically < 0.1 rad (5.7°) considered excellent")
    print("  • Your mean SAD: 0.717 rad (~41.1°) - moderate accuracy")
    
    print("\nRoot Mean Square Error (RMSE):")
    print("  • Measures abundance estimation accuracy")
    print("  • Lower values indicate better abundance estimation")
    print("  • Values closer to 0 are better, depends on scale of abundances")
    print("  • Your mean RMSE: 0.379 - moderate accuracy, could be improved")
    
    print("\nOverall Assessment:")
    print("  • Model successfully learned to estimate abundances for 3 classes")
    print("  • Room for improvement in both spectral preservation and abundance estimation")
    print("  • Consider adjusting hyperparameters or training longer for better results")

def create_unique_visualizations():
    """
    创建独特的可视化图表，避免与他人论文重复
    """
    # 从你的训练结果中获取的数据
    class_labels = ['Vegetation', 'Soil', 'Water']
    sad_values = [0.550, 0.907, 0.692]  # 从你的结果
    rmse_values = [0.379, 0.389, 0.370]  # 从你的结果
    
    # 创建一个新的图形布局
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 堆叠柱状图 - 展示SAD和RMSE的组合
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(class_labels))
    ax1.bar(x_pos, sad_values, label='SAD', alpha=0.7, color='lightcoral')
    ax1.bar(x_pos, rmse_values, bottom=sad_values, label='RMSE', alpha=0.7, color='lightblue')
    ax1.set_xlabel('Land Cover Types')
    ax1.set_ylabel('Value')
    ax1.set_title('Stacked Bar Chart: SAD & RMSE\nper Land Cover Type', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_labels, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 雷达图 - 不同于传统的雷达图
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    # 为SAD和RMSE创建独立的雷达图
    sad_values_closed = sad_values + [sad_values[0]]
    rmse_values_closed = rmse_values + [rmse_values[0]]
    
    ax2.plot(angles, sad_values_closed, 'o-', linewidth=2, label='SAD', color='red')
    ax2.fill(angles, sad_values_closed, alpha=0.25, color='red')
    ax2.plot(angles, rmse_values_closed, 's-', linewidth=2, label='RMSE', color='blue')
    ax2.fill(angles, rmse_values_closed, alpha=0.25, color='blue')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(['Vegetation', 'Soil', 'Water'])
    ax2.set_title('Polar Plot Comparison\nSAD vs RMSE', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 3. 散点图 - SAD vs RMSE
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(sad_values, rmse_values, c=range(len(class_labels)), 
                         cmap='viridis', s=200, alpha=0.7, edgecolors='black')
    for i, txt in enumerate(class_labels):
        ax3.annotate(txt, (sad_values[i], rmse_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    ax3.set_xlabel('Spectral Angle Distance (SAD)')
    ax3.set_ylabel('Root Mean Square Error (RMSE)')
    ax3.set_title('SAD vs RMSE Scatter Plot\nPer Land Cover Type', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 添加对角线参考线
    ax3.plot([0, max(sad_values + rmse_values)], [0, max(sad_values + rmse_values)], 
             'k--', alpha=0.5, label='Perfect Correlation')
    ax3.legend()
    
    # 4. 面积图 - 展示性能趋势
    ax4 = plt.subplot(2, 3, 4)
    x_classes = np.arange(len(class_labels))
    ax4.fill_between(x_classes, 0, sad_values, alpha=0.5, label='SAD', color='red')
    ax4.fill_between(x_classes, sad_values, np.array(sad_values) + np.array(rmse_values), 
                     alpha=0.5, label='RMSE', color='blue')
    ax4.set_xlabel('Land Cover Types')
    ax4.set_ylabel('Cumulative Metric Value')
    ax4.set_title('Area Chart: Cumulative\nPerformance Metrics', fontweight='bold')
    ax4.set_xticks(x_classes)
    ax4.set_xticklabels(class_labels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 气泡图 - 结合SAD和RMSE
    ax5 = plt.subplot(2, 3, 5)
    bubble_sizes = [(s + r) * 1000 for s, r in zip(sad_values, rmse_values)]  # 气泡大小基于两者的和
    scatter2 = ax5.scatter(range(len(class_labels)), sad_values, 
                          s=bubble_sizes, alpha=0.5, c=rmse_values, 
                          cmap='coolwarm', edgecolors='black')
    for i, txt in enumerate(class_labels):
        ax5.annotate(txt, (i, sad_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax5.set_xlabel('Land Cover Types')
    ax5.set_ylabel('Spectral Angle Distance (SAD)')
    ax5.set_title('Bubble Chart: SAD with Bubble Size\nProportional to Total Error', fontweight='bold')
    cbar = plt.colorbar(scatter2)
    cbar.set_label('RMSE Value')
    ax5.set_xticks(range(len(class_labels)))
    ax5.set_xticklabels(class_labels)
    ax5.grid(alpha=0.3)
    
    # 6. 3D散点图
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    from mpl_toolkits.mplot3d import Axes3D
    z_values = np.zeros(len(class_labels))  # Z轴可以代表其他指标或保持为0
    ax6.scatter(sad_values, rmse_values, z_values, c=range(len(class_labels)), 
               cmap='tab10', s=200, alpha=0.7)
    
    for i, txt in enumerate(class_labels):
        ax6.text(sad_values[i], rmse_values[i], z_values[i], txt, 
                fontsize=10, fontweight='bold')
    
    ax6.set_xlabel('SAD')
    ax6.set_ylabel('RMSE')
    ax6.set_zlabel('Index')
    ax6.set_title('3D Scatter Plot\nSAD vs RMSE', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('trans_mod_samson/unique_samson_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    dataset = 'samson'  # 修改这里来处理不同的数据集
    
    # 确保输出目录存在
    os.makedirs(f"trans_mod_{dataset}", exist_ok=True)
    
    # 创建指标对比图
    plot_training_metrics(dataset)
    
    # 创建详细指标图
    plot_detailed_metrics(dataset)
    
    # 可视化结果
    visualize_samson_results()
    
    # 创建独特可视化
    create_unique_visualizations()
    
    # 解释指标
    interpret_metrics()
    
    print(f"\nMetrics visualization completed for {dataset} dataset!")
    print("Check the generated PNG files in the trans_mod_samson folder.")