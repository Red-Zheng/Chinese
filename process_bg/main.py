# from PIL import Image
# import os
# import random

# # 田字格和米字格图片的路径
# tianzige_path = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/tianzige.jpg"
# mizige_path = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/mizige.jpg"

# # 读取田字格和米字格图片
# tianzige = Image.open(tianzige_path)
# mizige = Image.open(mizige_path)

# # 定义一个函数来调整背景图片的尺寸
# def resize_background(background, target_width, target_height):
#     return background.resize((target_width, target_height), Image.Resampling.LANCZOS)

# # 定义一个函数来将白色背景转换为透明背景
# def make_background_transparent(image):
#     # 将图片转换为RGBA模式
#     image = image.convert("RGBA")
#     # 获取图片的像素数据
#     data = image.getdata()
#     new_data = []
#     for item in data:
#         # 如果像素是白色（255, 255, 255），则将其设置为透明
#         if item[:3] == (255, 255, 255):
#             new_data.append((255, 255, 255, 0))
#         else:
#             new_data.append(item)
#     # 更新图片的像素数据
#     image.putdata(new_data)
#     return image

# # 定义一个函数来合成图片
# def composite_image(char_image, background_image):
#     # 将汉字图片的白色背景转换为透明背景
#     char_image = make_background_transparent(char_image)
    
#     # 调整背景图片的尺寸以匹配汉字图片
#     background_image = resize_background(background_image, char_image.width, char_image.height)
    
#     # 创建一个新的RGBA图像，尺寸与汉字图片相同
#     composite = Image.new("RGBA", char_image.size, (255, 255, 255, 0))
    
#     # 将背景图片粘贴到新的RGBA图像上
#     composite.paste(background_image, (0, 0))
    
#     # 将汉字图片粘贴到新的RGBA图像上，使用汉字图片的alpha通道作为掩码
#     composite.paste(char_image, (0, 0), char_image)
    
#     return composite
# # 处理文件夹中的所有汉字图片
# input_folder = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/val2021"
# output_folder = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/val2021_bg"

# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # 读取汉字图片
#         char_image = Image.open(os.path.join(input_folder, filename))
        
#         # 随机选择田字格或米字格作为背景
#         if random.choice([True, False]):
#             background_image = tianzige
#         else:
#             background_image = mizige
        
#         # 合成图片
#         composite = composite_image(char_image, background_image)
        
#         # 将RGBA模式转换为RGB模式，以便保存为JPEG格式
#         composite = composite.convert("RGB")
        
#         # 保存合成后的图片
#         output_path = os.path.join(output_folder, filename)
#         composite.save(output_path)
#         print(f"Saved {output_path}")

# print("All images processed.")


from PIL import Image
import os
import random
import time

# 田字格和米字格图片的路径
tianzige_path = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/tianzige.jpg"
mizige_path = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/mizige.jpg"

# 读取田字格和米字格图片
tianzige = Image.open(tianzige_path)
mizige = Image.open(mizige_path)

# 定义一个函数来调整背景图片的尺寸
def resize_background(background, target_width, target_height):
    return background.resize((target_width, target_height), Image.Resampling.LANCZOS)

# 定义一个函数来将白色背景转换为透明背景
def make_background_transparent(image):
    # 将图片转换为RGBA模式
    image = image.convert("RGBA")
    # 获取图片的像素数据
    data = image.getdata()
    new_data = []
    for item in data:
        # 如果像素是白色（255, 255, 255），则将其设置为透明
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    # 更新图片的像素数据
    image.putdata(new_data)
    return image

# 定义一个函数来合成图片
def composite_image(char_image, background_image):
    # 将汉字图片的白色背景转换为透明背景
    char_image = make_background_transparent(char_image)
    
    # 调整背景图片的尺寸以匹配汉字图片
    background_image = resize_background(background_image, char_image.width, char_image.height)
    
    # 创建一个新的RGBA图像，尺寸与汉字图片相同
    composite = Image.new("RGBA", char_image.size, (255, 255, 255, 0))
    
    # 将背景图片粘贴到新的RGBA图像上
    composite.paste(background_image, (0, 0))
    
    # 将汉字图片粘贴到新的RGBA图像上，使用汉字图片的alpha通道作为掩码
    composite.paste(char_image, (0, 0), char_image)
    
    return composite

# 处理文件夹中的所有汉字图片
input_folder = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/val2021_3"
output_folder = "/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/process_bg/val2021_bg"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]
total_images = len(image_files)

# 开始处理
start_time = time.time()
for i, filename in enumerate(image_files):
    # 读取汉字图片
    char_image = Image.open(os.path.join(input_folder, filename))
    
    # 随机选择田字格或米字格作为背景
    if random.choice([True, False]):
        background_image = tianzige
    else:
        background_image = mizige
    
    # 合成图片
    composite = composite_image(char_image, background_image)
    
    # 将RGBA模式转换为RGB模式，以便保存为JPEG格式
    composite = composite.convert("RGB")
    
    # 保存合成后的图片
    output_path = os.path.join(output_folder, filename)
    composite.save(output_path)
    
    # 打印进度
    print(f"Processed {i + 1}/{total_images}: {filename} -> Saved {output_path}")

# 处理完成
end_time = time.time()
print(f"All images processed in {end_time - start_time:.2f} seconds.")