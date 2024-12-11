import math
import os
import cv2
import numpy as np
import tifffile as tiff
import tqdm
from django.conf import settings
from utils.funcs import Fun_Spectral_Correction, Fun_All_Factors2
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
import redis
# 初始化 Redis 连接
redis_instance = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)
def soil_remove(task_id, path):
    task_type = 'type1'
    total_steps = 100
    step = 0
    # 读取 TIFF 图像
    filename = path
    T1 = tiff.imread(filename).astype(float) / 10000  # 读取图像并进行归一化
    T1 = T1.transpose((1, 2, 0))
    # A 即为 T1
    A = T1
    # 获取 A 的尺寸
    m, n, k = A.shape
    Ar = A[:, :, 2]
    Ag = A[:, :, 1]
    Ab = A[:, :, 0]
    Are = A[:, :, 3]
    Anir = A[:, :, 4]

    div = Anir-Ar

    ck = 25

    m0 = m - m % ck
    n0 = n - n % ck
    # div_record = [m0 / ck, n0 / ck]

    # 初始化用于存储植被密集度等级划分的矩阵
    div_record = np.zeros((int(m0 / ck), int(n0 / ck)))

    with tqdm.tqdm(total=int(m0 / ck), desc="处理中") as pbar:
        # 循环遍历每个分块
        for i in range(int(m0 / ck)):
            for j in range(int(n0 / ck)):
                h = ck - (ck - 1) / 2 + i * ck
                l = ck - (ck - 1) / 2 + j * ck
                w = int(h - (ck - 1) / 2)
                e = int(l - (ck - 1) / 2)

                # 提取当前窗口的分块数据
                window_div = np.zeros((ck, ck))
                for z in range(ck):
                    # print(div[w + z, e:e + ck])
                    window_div[z, :] = div[w - 1 + z, e - 1:e - 1 + ck]

                # 计算窗口的平均值
                mean_div = np.mean(window_div)
                div_record[i, j] = mean_div

            # 更新进度条
            pbar.update(1)

        # 植被密集度等级划分
    _, binEdges = np.histogram(div_record.flatten(), bins=5)

    # 自适应光谱增强与图像阈值分割
    shape_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    W_record = np.zeros((m0, n0))

    step += 10
    progress = step / total_steps
    redis_instance.set(f"task_progress_{task_id}", progress*100)


    # 创建进度条
    with tqdm.tqdm(total=int(m0 / ck), desc="图像处理中") as pbar:
        for i in range(int(m0 / ck)):
            for j in range(int(n0 / ck)):
                # 计算窗口的起始位置
                h = int(ck - (ck - 1) / 2 + i * ck)
                l = int(ck - (ck - 1) / 2 + j * ck)
                w = int(h - (ck - 1) / 2)
                e = int(l - (ck - 1) / 2)

                window_Ar = Ar[w - 1:w - 1 + ck, e - 1:e - 1 + ck]
                window_Anir = Anir[w - 1:w - 1 + ck, e - 1:e - 1 + ck]

                # 计算 mean_div
                mean_div = div_record[i, j]

                # 根据 mean_div 确定 p0 的值
                if binEdges[0] < mean_div <= binEdges[1]:
                    p0 = 0.4
                elif binEdges[1] < mean_div <= binEdges[2]:
                    p0 = 0.35
                elif binEdges[2] < mean_div <= binEdges[3]:
                    p0 = 0.3
                elif binEdges[3] < mean_div <= binEdges[4]:
                    p0 = 0.25
                elif binEdges[4] < mean_div <= binEdges[5]:
                    p0 = 0.2

                # 自适应光谱增强
                Anir_unique, l_idx = np.unique(window_Anir, return_inverse=True)
                Anir_sort = np.argsort(Anir_unique)
                T = len(Anir_sort)
                p, q, h = p0, 0.6, 0.15

                Fa = Fun_Spectral_Correction(T, p, q, h, l_idx, Anir_sort, ck, ck)
                window_Anir_adj = window_Anir * Fa

                # 图像阈值分割
                R = window_Anir_adj - window_Ar
                R_ad = exposure.equalize_adapthist(R)  # 自适应直方图均衡化
                # R_ad = adapthistep(R)
                scaler = MinMaxScaler((0, 1))
                R_ad = scaler.fit_transform(R_ad.T).T
                rm_ad = np.mean(R_ad)

                NDVI_adj = (window_Anir_adj - window_Ar) / (window_Anir_adj + window_Ar)
                maxndvi = NDVI_adj.max()
                minndvi = NDVI_adj.min()
                meanndvi = np.mean(NDVI_adj)
                cv = ((meanndvi - minndvi) / (maxndvi - meanndvi)) ** 2

                if cv > 1:
                    c = cv
                    c_a = 2 * c * (10 ** -2)
                    l = rm_ad + c_a
                else:
                    c = -cv
                    c_a = 2 * (1 / c) * (10 ** -2)
                    l = rm_ad + c_a

                    # 根据阈值 l 进行处理
                W = np.zeros((ck, ck))
                for e_idx in range(ck):
                    for f_idx in range(ck):
                        if R_ad[e_idx, f_idx] > l:
                            W[e_idx, f_idx] = window_Anir_adj[e_idx, f_idx] + 0.1
                        else:
                            W[e_idx, f_idx] = window_Anir_adj[e_idx, f_idx] - 0.1

                # 将 W 归一化到 0-1
                W_01 = scaler.fit_transform(W.T).T

                # 根据 W_01 值调整 Ar_w
                Ar_w = window_Ar.copy()
                for e_idx in range(ck):
                    for f_idx in range(ck):
                        if W_01[e_idx, f_idx] < 0.5:
                            Ar_w[e_idx, f_idx] = 0  # red
                        else:
                            Ar_w[e_idx, f_idx] = 1  # red

                # 计算形状特征
                Mask = np.where(Ar_w == 0)
                Mask_num = len(Mask[0])
                Mask_rate = Mask_num / (ck * ck)
                Green_rate = 1 - Mask_rate
                shape_record[i, j] = Green_rate

                # 更新 W_record
                W_record[i * ck:(i + 1) * ck, j * ck:(j + 1) * ck] = Ar_w

            step += 90 / int(m0 / ck)
            progress = step / total_steps
            redis_instance.set(f"task_progress_{task_id}", progress * 100)

            # 更新进度条
            pbar.update(1)

    W_record_uint8 = (W_record * 255).astype(np.uint8)

    # 构造保存路径
    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    file_name = f'W_record_saved_{task_id}.png'
    file_path = os.path.join(task_results_folder, file_name)
    cv2.imwrite(file_path, W_record_uint8)

    redis_instance.set(f"task_progress_{task_id}", 100)

    # # cv2.imshow('W_record', W_record_uint8)
    #
    # # 掩膜应用
    # masked_image = A[:m0, :n0, :] * W_record[:, :, np.newaxis]
    # # 确保数据类型匹配，以与原始图像 A 的数据类型一致
    # masked_image = masked_image.astype(A[:m0, :n0, :].dtype)
    #
    # # 提取指数
    # outmat = Fun_All_Factors2(masked_image, m0, n0, ck)
    #
    # # 将 shape_record 展平
    # shape_record1 = shape_record.flatten()
    #
    # # 将 outmat 和 shape_record1 进行水平拼接
    # all_factors = np.hstack((outmat, shape_record1[:, np.newaxis]))

    return f'W_record_saved_{task_id}.png'
