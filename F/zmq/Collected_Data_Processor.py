# author: zmq 2024.12.3
import os
import random
import re
import string
import redis
import tqdm
from django.conf import settings
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from skimage import exposure
import rasterio
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils.funcs import Fun_Spectral_Correction
import scipy

redis_instance = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)
def dms_to_decimal(degrees, minutes, seconds):
    """
    将度、分、秒（DMS）转换为十进制度数（Decimal Degrees）
    """
    return degrees + (minutes / 60) + (seconds / 3600)
def calculate_features(A, y_pixel, x_pixel):
    A = np.array(A).transpose((1, 2, 0))
    y_pixel = int(y_pixel)
    x_pixel = int(x_pixel)
    ck = 25
    Ar = A[:, :, 2]  # 第三波段 (红光)
    Ag = A[:, :, 1]  # 第二波段 (绿光)
    Ab = A[:, :, 0]  # 第一波段 (蓝光)
    Are = A[:, :, 3]  # 第四波段 (红边)
    Anir = A[:, :, 4]  # 第五波段 (近红外)
    gray_Anir = exposure.rescale_intensity(Anir, out_range=(0, 1))

    # 提取窗口
    window_Ar = Ar[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]
    window_Ag = Ag[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]
    window_Ab = Ab[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]
    window_Are = Are[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]
    window_Anir = Anir[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]

    # 计算颜色矩阵
    R = np.mean(window_Ar)
    G = np.mean(window_Ag)
    Rstd = np.std(window_Ar)
    Gstd = np.std(window_Ag)
    Bstd = np.std(window_Ab)
    Rmean = R
    Gmean = G
    Bmean = np.mean(window_Ab)

    # 植被指数计算

    NIR = np.mean(window_Anir)
    RE = np.mean(window_Are)
    NDVI = (NIR - R) / (NIR + R)
    EVI2 = 2.5 * (NIR - R) / (1 + NIR + 2.4 * R)
    RVI = NIR / R
    DVI = NIR - R
    RDVI = (NIR - R) / np.sqrt(NIR + R)
    MSR = (NIR / R - 1) / np.sqrt(NIR / R + 1)
    MCARI = (NIR - RE - 0.2 * (NIR - G)) / (NIR / RE)
    OSAVI = (1.16 * (NIR - R)) / (NIR + R + 0.16)
    WDRVI = (0.1 * NIR - R) / (0.1 * NIR + R)

    # 纹理指数计算
    window_gray_Anir = gray_Anir[y_pixel - ck:y_pixel, x_pixel - ck:x_pixel]
    image_uint8 = (window_gray_Anir * 63).astype(np.uint8)
    glcms1 = graycomatrix(image_uint8, levels=64, distances=[1],
                          angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    ga1 = glcms1[:, :, 0, 0]
    ga2 = glcms1[:, :, 0, 1]
    ga3 = glcms1[:, :, 0, 2]
    ga4 = glcms1[:, :, 0, 3]
    energya1 = 0
    energya2 = 0
    energya3 = 0
    energya4 = 0
    homogeneity1 = 0
    homogeneity2 = 0
    homogeneity3 = 0
    homogeneity4 = 0

    for e in range(64):
        for f in range(64):
            energya1 = energya1 + ga1[e, f] ** 2
            homogeneity1 = homogeneity1 + (1 / (1 + (e - f) ** 2)) * ga1[e, f]
            energya2 = energya2 + ga2[e, f] ** 2
            homogeneity2 = homogeneity2 + (1 / (1 + (e - f) ** 2)) * ga2[e, f]
            energya3 = energya3 + ga3[e, f] ** 2
            homogeneity3 = homogeneity3 + (1 / (1 + (e - f) ** 2)) * ga3[e, f]
            energya4 = energya4 + ga4[e, f] ** 2
            homogeneity4 = homogeneity4 + (1 / (1 + (e - f) ** 2)) * ga4[e, f]
    s1 = np.sum(graycoprops(glcms1, 'contrast'))
    s2 = np.sum(graycoprops(glcms1, 'correlation'))
    s3 = np.sum(graycoprops(glcms1, 'energy'))
    s4 = np.sum(graycoprops(glcms1, 'homogeneity'))
    s5 = 0.00001 * (energya1 + energya2 + energya3 + energya4)
    s6 = 0.0001 * (homogeneity1 + homogeneity2 + homogeneity3 + homogeneity4)
    S1 = s1
    S2 = s2
    S3 = s3
    S4 = s4
    S5 = s5
    S6 = s6

    # 形状特征
    div = Anir - Ar
    # 提取当前窗口的分块数据
    window_div = np.zeros((ck, ck))
    for z in range(ck):
        # print(div[w + z, e:e + ck])
        window_div[z, :] = div[y_pixel + z - ck, x_pixel - ck:x_pixel]

    # 计算窗口的平均值
    mean_div = np.mean(window_div)
    div_record = mean_div
    _, binEdges = np.histogram(div_record, bins=5)

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
    else:
        p0 = 1

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

    NDVI_adj = window_Anir_adj - window_Ar / window_Anir_adj + window_Ar

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
    W = np.zeros((ck, ck))
    for e_idx in range(ck):
        for f_idx in range(ck):
            if R_ad[e_idx, f_idx] > l:
                W[e_idx, f_idx] = window_Anir_adj[e_idx, f_idx] + 0.1
            else:
                W[e_idx, f_idx] = window_Anir_adj[e_idx, f_idx] - 0.1

    # 将 W 归一化到 0-1
    W_01 = scaler.fit_transform(W.T).T

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
    shape_record = Green_rate

    outmat = np.column_stack((NDVI, EVI2, RVI, DVI, RDVI, MSR,
                              MCARI, OSAVI, WDRVI, NIR, S1, S2, S3, S4,
                              S5, S6, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd, shape_record))
    return outmat
# 打开GeoTIFF文件
def get_pixel_value_from_geolocation(tif_file, lon, lat):
    with rasterio.open(tif_file) as dataset:
        # 获取GeoTIFF的坐标参考系统（CRS）和仿射变换矩阵
        crs = dataset.crs
        transform = dataset.transform

        # 检查CRS是否为WGS84 (EPSG:4326)，如果是，则直接进行转换
        if crs.to_string() == 'EPSG:4326':  # WGS84经纬度坐标系
            # 将经纬度 (lon, lat) 转换为图像的像素坐标
            x_pixel, y_pixel = ~transform * (lon, lat)
            # print(f"Pixel coordinates: (x: {x_pixel}, y: {y_pixel})")

            # 获取像素的值，注意y_pixel的顺序是倒序的（从上到下）
            row, col = int(y_pixel), int(x_pixel)
            # print(f"Picture size: {dataset.read(1).shape}")
            # 获取所有波段的像素值
            pixel_values = []
            bands = []
            for band in range(1, dataset.count + 1):  # dataset.count 返回波段数量
                band_data = dataset.read(band)
                pixel_values.append(band_data[row, col])
                bands.append(band_data)
            outmat = calculate_features(bands, x_pixel=x_pixel, y_pixel=y_pixel)
            return outmat
            # # 输出各波段的像素值
            # print(f"Pixel values at (lon: {lon}, lat: {lat}):")
            # for i, pixel_value in enumerate(pixel_values, start=1):
            #     print(f"  Band {i}: {pixel_value}")
            # print(outmat)
        else:
            print("CRS is not EPSG:4326. Please reproject the image to EPSG:4326 for longitude/latitude coordinates.")


def dms_to_tuple(dms_str):
    # 使用正则表达式从字符串中提取度、分、秒
    match = re.match(r'(\d+)°(\d+)′([\d.]+)″[NSEW]', dms_str)
    if match:
        # 提取度、分、秒的值
        degrees = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))

        # 格式化秒值为一定的小数位数
        seconds = round(seconds, 20)  # 可以调整精度

        # 返回元组
        return (degrees, minutes, seconds)
    else:
        raise ValueError("输入的字符串格式不正确")

def collected_data_processor(task_id, files, appendmat=False):
    redis_instance.set(f"task_progress_{task_id}", 20)
    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    file_name = 'featuresPlusTargetMat_' + random_str + '.mat'
    output_file = os.path.join(task_results_folder, file_name)
    column_names = ['Column1', 'Column2', 'Column3', 'Column4']
    df = pd.read_csv(files[0], encoding='gbk', names=column_names)

    tif_file = files[1]  # 你的GeoTIFF文件路径

    # 提取某一列
    lats = np.array(df['Column2'])  # 使用列名提取数据
    lons = np.array(df['Column3'])
    targets = np.array(df['Column4'])
    lats = [dms_to_tuple(item) for item in lats]
    lons = [dms_to_tuple(item) for item in lons]
    targets = [float(item) for item in targets]
    total_num = len(lats)
    features = np.empty((total_num, 23))
    with tqdm.tqdm(total=int(total_num), desc="处理中") as pbar:
        state = 20
        for i in range(total_num):
            # DMS格式经纬度
            latitude_dms = lats[i]  # 纬度：36°51′08.4979764000″N
            longitude_dms = lons[i]  # 经度：115°00′34.7184594000″E

            # 转换为十进制度数
            latitude_decimal = dms_to_decimal(*latitude_dms)
            longitude_decimal = dms_to_decimal(*longitude_dms)

            # print(f"Latitude in decimal degrees: {latitude_decimal}")
            # print(f"Longitude in decimal degrees: {longitude_decimal}")
            features[i, :] = np.array(get_pixel_value_from_geolocation(tif_file, longitude_decimal, latitude_decimal))
            state += 80/int(total_num)
            redis_instance.set(f"task_progress_{task_id}", state)
            # 更新进度条
            pbar.update(1)
    featuresPlusTargetMat = np.hstack((features, np.array(targets)[:, np.newaxis]))

    if appendmat:
        old_mat = scipy.io.loadmat(files[2])
        old_mat['featuresPlusTargetMat'] = np.concatenate((old_mat['featuresPlusTargetMat'], featuresPlusTargetMat))
        scipy.io.savemat(output_file, old_mat)
    else:
        scipy.io.savemat(output_file, {'featuresPlusTargetMat': featuresPlusTargetMat})
    redis_instance.set(f"task_progress_{task_id}", 100)
    return [file_name]
