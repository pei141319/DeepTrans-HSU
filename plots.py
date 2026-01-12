# import numpy as np
# from matplotlib import pyplot as plt


# def plot_abundance(ground_truth, estimated, em, save_dir):

#     plt.figure(figsize=(12, 6), dpi=150)
#     for i in range(em):
#         plt.subplot(2, em, i + 1)
#         plt.imshow(ground_truth[:, :, i], cmap='jet')

#     for i in range(em):
#         plt.subplot(2, em, em + i + 1)
#         plt.imshow(estimated[:, :, i], cmap='jet')
#     plt.tight_layout()

#     plt.savefig(save_dir + "abundance.png")


# def plot_endmembers(target, pred, em, save_dir):

#     plt.figure(figsize=(12, 6), dpi=150)
#     for i in range(em):
#         plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
#         plt.plot(pred[:, i], label="Extracted")
#         plt.plot(target[:, i], label="GT")
#         plt.legend(loc="upper left")
#     plt.tight_layout()

#     plt.savefig(save_dir + "end_members.png")

import numpy as np
from matplotlib import pyplot as plt


def plot_abundance(ground_truth, estimated, em, save_dir, dataset='unknown'):

    plt.figure(figsize=(12, 6), dpi=150)
    # 根据数据集类型定义地物类别名称
    if dataset == 'samson':
        class_names = ['Soil', 'Tree', 'Water']
    elif dataset == 'jasper':
        class_names = ['Veg', 'Soil', 'Water', 'Road']
    elif dataset == 'apex':
        # APEX数据集的标准4端元：水体、植被、道路、土壤
        class_names = ['Water', 'Tree', 'Road', 'Soil']
    else:
        class_names = [f'Endmember {i + 1}' for i in range(em)]
    
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')
        plt.title(f'GT {class_names[i]}')

    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
        plt.title(f'Est {class_names[i]}')
    plt.tight_layout()

    plt.savefig(save_dir + "abundance.png")


def plot_endmembers(target, pred, em, save_dir, dataset='unknown'):

    plt.figure(figsize=(12, 6), dpi=150)
    # 根据数据集类型定义地物类别名称
    if dataset == 'samson':
        class_names = ['Soil', 'Tree', 'Water']
    elif dataset == 'jasper':
        class_names = ['Veg', 'Soil', 'Water', 'Road']
    else:
        class_names = [f'Endmember {i + 1}' for i in range(em)]
        
    for i in range(em):
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i], label="Extracted")
        plt.plot(target[:, i], label="GT")
        plt.title(class_names[i])
        plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_dir + "end_members.png")