import random
import string

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import redis
from django.conf import settings

redis_instance = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)
def read_band(filename):
    with rasterio.open(filename) as src:
        result = np.array(src.read(1))
        result[np.isnan(result)] = 0
        return result

def read_band_with_info(filename):
    with rasterio.open(filename) as src:
        crs = src.crs  # 获取坐标参考系统（CRS）
        transform = src.transform  # 获取空间变换
        width = src.width  # 图像宽度
        height = src.height  # 图像高度
    return crs, transform, width, height

def picture_synthesis(task_id, files):
    redis_instance.set(f"task_progress_{task_id}", 15)
    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    file_name = 'composite_image_' + random_str + '.tif'
    output_file = os.path.join(task_results_folder, file_name)
    # print(output_file)
    layers = [read_band(file) for file in files]
    crs, transform, width, height = read_band_with_info(files[0])
    redis_instance.set(f"task_progress_{task_id}", 65)
    # 合成多波段图像
    multi_band_image = np.stack(layers, axis=-1)
    # 保存合成后的多波段影像
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       count=len(files),  # 5个波段
                       width=multi_band_image.shape[1],
                       height=multi_band_image.shape[0],
                       dtype=multi_band_image.dtype,
                       crs=crs,  # 使用适当的 CRS (例如 WGS84)
                       transform=transform  # 示例：适当的空间变换
                       ) as dst:
        for i in range(5):
            dst.write(multi_band_image[:, :, i], i + 1)  # 将每个波段写入文件
    redis_instance.set(f"task_progress_{task_id}", 100)
    return [file_name]