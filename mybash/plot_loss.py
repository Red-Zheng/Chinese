# import argparse
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# def parse_args():
#     parser = argparse.ArgumentParser(description='Plot mean loss per epoch from JSON file')
#     parser.add_argument('json_path', type=str, help='Path to the JSON file')
#     parser.add_argument('output_path', type=str, help='Path to save the output image')
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # 加载JSON文件
#     data = []
#     with open(args.json_path, 'r') as f:
#         for line in f:
#             entry = json.loads(line)
#             data.append(entry)

#     # 提取每个epoch的损失值
#     epochs = []
#     losses = []

#     for entry in data:
#         if 'epoch' in entry and 'loss' in entry:
#             epoch = entry['epoch']
#             loss = entry['loss']
#             epochs.append(epoch)
#             losses.append(loss)

#     # 计算每个epoch的平均损失值
#     unique_epochs = np.unique(epochs)
#     mean_losses = [np.mean([losses[i] for i in range(len(losses)) if epochs[i] == epoch]) for epoch in unique_epochs]

#     print(mean_losses)

#     # 绘制图表
#     plt.plot(unique_epochs, mean_losses, marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(False)

#     # 保存图表为图片
#     plt.savefig(args.output_path)

# if __name__ == '__main__':
#     main()

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Plot mean loss per epoch from JSON file')
    parser.add_argument('json_path', type=str, help='Path to the JSON file')
    parser.add_argument('output_path', type=str, help='Path to save the output image')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载JSON文件
    data = []
    with open(args.json_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)

    # 提取每个epoch的损失值
    epochs = []
    s0_loss_bbox = []
    s0_loss_mask = []
    s1_loss_bbox = []
    s1_loss_mask = []
    s2_loss_bbox = []
    s2_loss_mask = []
    total_loss = []

    for entry in data:
        if 'epoch' in entry:
            epoch = entry['epoch']
            epochs.append(epoch)
            s0_loss_bbox.append(entry.get('s0.loss_bbox_reg', 0))
            s0_loss_mask.append(entry.get('s0.loss_mask', 0))
            s1_loss_bbox.append(entry.get('s1.loss_bbox_reg', 0))
            s1_loss_mask.append(entry.get('s1.loss_mask', 0))
            s2_loss_bbox.append(entry.get('s2.loss_bbox_reg', 0))
            s2_loss_mask.append(entry.get('s2.loss_mask', 0))
            total_loss.append(entry.get('loss', 0))

    # 计算每个epoch的平均损失值
    unique_epochs = np.unique(epochs)
    mean_s0_loss_bbox = [np.mean([s0_loss_bbox[i] for i in range(len(s0_loss_bbox)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_s0_loss_mask = [np.mean([s0_loss_mask[i] for i in range(len(s0_loss_mask)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_s1_loss_bbox = [np.mean([s1_loss_bbox[i] for i in range(len(s1_loss_bbox)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_s1_loss_mask = [np.mean([s1_loss_mask[i] for i in range(len(s1_loss_mask)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_s2_loss_bbox = [np.mean([s2_loss_bbox[i] for i in range(len(s2_loss_bbox)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_s2_loss_mask = [np.mean([s2_loss_mask[i] for i in range(len(s2_loss_mask)) if epochs[i] == epoch]) for epoch in unique_epochs]
    mean_total_loss = [np.mean([total_loss[i] for i in range(len(total_loss)) if epochs[i] == epoch]) for epoch in unique_epochs]

    # 绘制四宫格图表
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 左上图：s0.loss_bbox_reg 和 s0.loss_mask
    axs[0, 0].plot(unique_epochs, mean_s0_loss_bbox, marker='o', label='s0.loss_bbox')
    axs[0, 0].plot(unique_epochs, mean_s0_loss_mask, marker='o', label='s0.loss_mask')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Stage 0 Losses')
    axs[0, 0].legend()
    axs[0, 0].grid(False)

    # 右上图：s1.loss_bbox_reg 和 s1.loss_mask
    axs[0, 1].plot(unique_epochs, mean_s1_loss_bbox, marker='o', label='s1.loss_bbox')
    axs[0, 1].plot(unique_epochs, mean_s1_loss_mask, marker='o', label='s1.loss_mask')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Stage 1 Losses')
    axs[0, 1].legend()
    axs[0, 1].grid(False)

    # 左下图：s2.loss_bbox_reg 和 s2.loss_mask
    axs[1, 0].plot(unique_epochs, mean_s2_loss_bbox, marker='o', label='s2.loss_bbox')
    axs[1, 0].plot(unique_epochs, mean_s2_loss_mask, marker='o', label='s2.loss_mask')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('Stage 2 Losses')
    axs[1, 0].legend()
    axs[1, 0].grid(False)

    # 右下图：total loss
    axs[1, 1].plot(unique_epochs, mean_total_loss, marker='o', label='Total Loss')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].set_title('Total Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(False)

    # 保存图表为图片
    plt.tight_layout()
    plt.savefig(args.output_path)

if __name__ == '__main__':
    main()