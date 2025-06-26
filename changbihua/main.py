# import json
# from collections import defaultdict

# # 加载 COCO 标注文件
# with open('/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/datasets/handwritten_chinese_stroke_2021/annotations/instances_train2021.json', 'r') as f:
#     coco_data = json.load(f)

# # 初始化统计字典
# category_area = defaultdict(int)  # 每个类别的总面积
# category_image_pixels = defaultdict(int)  # 包含该类别的图片的总像素数

# # 遍历所有图片
# for image in coco_data['images']:
#     image_id = image['id']
#     image_width = image['width']
#     image_height = image['height']
#     image_pixels = image_width * image_height

#     # 遍历所有标注，找到属于当前图片的标注
#     for annotation in coco_data['annotations']:
#         if annotation['image_id'] == image_id:
#             category_id = annotation['category_id']
#             area = annotation['area']
#             category_area[category_id] += area
#             category_image_pixels[category_id] += image_pixels

# # 计算每个类别的面积占比
# category_ratio = {}
# for category_id in category_area:
#     if category_image_pixels[category_id] > 0:
#         ratio = category_area[category_id] / category_image_pixels[category_id]
#         category_ratio[category_id] = ratio

# # 将类别名称和面积占比组合成列表，并按面积占比从大到小排序
# sorted_categories = sorted(
#     coco_data['categories'],
#     key=lambda x: category_ratio.get(x['id'], 0),
#     reverse=True
# )

# # 输出结果（从大到小）
# for category in sorted_categories:
#     category_id = category['id']
#     category_name = category['name']
#     ratio = category_ratio.get(category_id, 0)
#     print(f"Category: {category_name}, Area Ratio: {ratio:.4f}")


# import json
# from collections import defaultdict
# import matplotlib.pyplot as plt

# # 加载 COCO 标注文件
# with open('/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/datasets/handwritten_chinese_stroke_2021/annotations/instances_train2021.json', 'r') as f:
#     coco_data = json.load(f)

# # 初始化统计字典
# category_area = defaultdict(int)  # 每个类别的总面积
# category_image_pixels = defaultdict(int)  # 包含该类别的图片的总像素数

# # 遍历所有图片
# for image in coco_data['images']:
#     image_id = image['id']
#     image_width = image['width']
#     image_height = image['height']
#     image_pixels = image_width * image_height

#     # 遍历所有标注，找到属于当前图片的标注
#     for annotation in coco_data['annotations']:
#         if annotation['image_id'] == image_id:
#             category_id = annotation['category_id']
#             area = annotation['area']
#             category_area[category_id] += area
#             category_image_pixels[category_id] += image_pixels

# # 计算每个类别的面积占比
# category_ratio = {}
# for category_id in category_area:
#     if category_image_pixels[category_id] > 0:
#         ratio = category_area[category_id] / category_image_pixels[category_id]
#         category_ratio[category_id] = ratio

# # 将类别名称和面积占比组合成列表，并按面积占比从大到小排序
# sorted_categories = sorted(
#     coco_data['categories'],
#     key=lambda x: category_ratio.get(x['id'], 0),
#     reverse=True
# )

# # 提取类别名称和面积占比
# category_names = []
# ratios = []
# for category in sorted_categories:
#     category_id = category['id']
#     category_name = category['name']
#     ratio = category_ratio.get(category_id, 0)
#     category_names.append(category_name)
#     ratios.append(ratio)

# # 可视化
# plt.figure(figsize=(10, 6))
# plt.barh(category_names, ratios, color='skyblue')
# plt.xlabel('Area Ratio')
# plt.ylabel('Category')
# plt.title('Area Ratio of Each Stroke Category')
# plt.gca().invert_yaxis()  # 从上到下显示，面积占比最大的在最上面
# plt.tight_layout()

# # 保存可视化结果
# output_path = 'stroke_category_area_ratio.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"可视化结果已保存到: {output_path}")

# # 显示可视化结果
# plt.show()


import json
from collections import defaultdict
import matplotlib.pyplot as plt

# 加载 COCO 标注文件
with open('/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/datasets/handwritten_chinese_stroke_2021/annotations/instances_train2021.json', 'r') as f:
    coco_data = json.load(f)

# 初始化统计字典
category_area = defaultdict(int)  # 每个类别的总面积
category_image_pixels = defaultdict(int)  # 包含该类别的图片的总像素数

# 定义长笔画的综合评分函数
def calculate_score(annotation, image_width):
    """
    计算笔画的综合评分，综合考虑长度、面积和长宽比。
    """
    bbox = annotation['bbox']  # [x, y, width, height]
    area = annotation['area']
    length = max(bbox[2], bbox[3])  # 长度：边界框的长边
    aspect_ratio = bbox[2] / bbox[3] if bbox[3] != 0 else 0  # 长宽比：width / height

    # 归一化处理
    normalized_length = length / image_width  # 长度归一化到图片宽度
    normalized_area = area / (image_width * image_width)  # 面积归一化到图片面积
    normalized_aspect_ratio = min(aspect_ratio, 10) / 10  # 长宽比归一化到 [0, 1]

    # 综合评分（权重可根据需求调整）
    score = 0.4 * normalized_length + 0.4 * normalized_area + 0.2 * normalized_aspect_ratio
    return score

# 初始化长笔画列表
long_strokes = []

# 遍历所有图片
for image in coco_data['images']:
    image_id = image['id']
    image_width = image['width']
    image_height = image['height']
    image_pixels = image_width * image_height

    # 遍历所有标注，找到属于当前图片的标注
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            area = annotation['area']
            category_area[category_id] += area
            category_image_pixels[category_id] += image_pixels

            # 计算综合评分
            score = calculate_score(annotation, image_width)

            # 判断是否为长笔画（阈值可根据需求调整）
            if score > 0.5:  # 假设阈值为 0.5
                long_strokes.append(annotation)

# 计算每个类别的面积占比
category_ratio = {}
for category_id in category_area:
    if category_image_pixels[category_id] > 0:
        ratio = category_area[category_id] / category_image_pixels[category_id]
        category_ratio[category_id] = ratio

# 将类别名称和面积占比组合成列表，并按面积占比从大到小排序
sorted_categories = sorted(
    coco_data['categories'],
    key=lambda x: category_ratio.get(x['id'], 0),
    reverse=True
)

# 提取类别名称和面积占比
category_names = []
ratios = []
for category in sorted_categories:
    category_id = category['id']
    category_name = category['name']
    ratio = category_ratio.get(category_id, 0)
    category_names.append(category_name)
    ratios.append(ratio)

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(category_names, ratios, color='skyblue')
plt.xlabel('Area Ratio')
plt.ylabel('Category')
plt.title('Area Ratio of Each Stroke Category')
plt.gca().invert_yaxis()  # 从上到下显示，面积占比最大的在最上面
plt.tight_layout()

# 保存可视化结果
output_path = 'stroke_category_area_ratio.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"可视化结果已保存到: {output_path}")

# 显示可视化结果
plt.show()
