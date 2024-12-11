import scipy
from scipy.io import loadmat
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops
import tqdm
from libsvm.svmutil import *
import cv2
from scipy.ndimage import generic_filter
def adapthistep(image, clip_limit=0.01, grid_size=(8, 8)):
    # 获取图像尺寸
    m, n = image.shape

    # 确定块大小
    grid_m, grid_n = grid_size
    block_m = m // grid_m
    block_n = n // grid_n

    # 创建输出图像
    result = np.zeros_like(image, dtype=np.float32)

    # 遍历每个块并进行直方图均衡化
    for i in range(grid_m):
        for j in range(grid_n):
            # 获取当前块的位置
            start_m = i * block_m
            end_m = (i + 1) * block_m if i < grid_m - 1 else m
            start_n = j * block_n
            end_n = (j + 1) * block_n if j < grid_n - 1 else n

            # 提取当前块
            block = image[start_m:end_m, start_n:end_n]

            # 计算直方图
            hist, bin_edges = np.histogram(block, bins=256, range=(0, 1))

            # 对比度剪裁
            excess_pixels = np.sum(hist) * clip_limit
            for k in range(len(hist)):
                if hist[k] > excess_pixels:
                    excess_pixels -= (hist[k] - excess_pixels)
                    hist[k] = excess_pixels

            # 重新计算剪裁后的 CDF
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]  # 归一化 CDF

            # 应用直方图均衡化
            block_mapped = np.interp(block.flatten(), bin_edges[:-1], cdf_normalized).reshape(block.shape)

            # 将均衡化后的块赋给输出图像
            result[start_m:end_m, start_n:end_n] = block_mapped

    # 归一化到 0-1 范围
    result = np.clip(result, 0, 1)

    return result

def column_to_row_major(matrix):
    # [[1 3 5]
    #  [2 4 6]]

    # [[1 2 3]
    #  [4 5 6]]
    # 获取矩阵的维度 (m x n)
    m, n = matrix.shape
    # 创建一个新的数组来存储转换后的数据 (行主序)
    row_major_matrix = np.zeros((m, n))
    arr = []
    # 将数据按照行主序填充
    for i in range(m):
        for j in range(n):
            # 在列主序中，第i行第j列的元素应该放到行主序中对应的位置
            temp = matrix[i, j]
            arr.append(temp)
    count = 0
    for i in range(m):
        for j in range(n):
            row_major_matrix[i, j] = arr[count]
            count = count + n
            if count >= n*m:
                count = count % (n*m) + 1
    return row_major_matrix

def Fun_Spectral_Correction(T, p, q, h, index, sort_order, m, n):
    ga = []
    t = 1
    while t <= T:
        if t <= p * T:
            a = 0
            ga.append(a)
        elif p * T < t <= T:
            t_q = (T - p * T) * q + p * T
            a = (-h / (p * T - t_q) ** 2) * (t - t_q) ** 2 + h
            if a < 0:
                a = 0
            ga.append(a)
        t += 1

    fa = 1 + np.array(ga)  # 直接使用numpy数组
    Fa = fa[sort_order]  # 排序
    Fa = Fa[index]  # 索引
    Fa = Fa.reshape(m, n)  # 重塑形状

    return Fa


def Fun_init(A, X_tr, Y_tr, X_te):
    X_tr = np.array(X_tr)
    Y_tr = np.array(Y_tr)
    X_te = np.array(X_te)

    P_train = A[X_tr, :]
    P_train = P_train[:, Y_tr].T

    T_train = A[X_tr, :]
    T_train = T_train[:, 23]
    T_train = T_train[np.newaxis, :]

    P_test = A[X_te, :]
    P_test = P_test[:, Y_tr].T

    T_test = A[X_te, :]
    T_test = T_test[:, 23]
    T_test = T_test[np.newaxis, :]

    # Initialize MinMaxScaler
    scaler_input = MinMaxScaler(feature_range=(0, 1))
    scaler_output = MinMaxScaler(feature_range=(0, 1))

    # Apply MinMaxScaler to P_train and T_train
    p_train = scaler_input.fit_transform(P_train.T)
    t_train = scaler_output.fit_transform(T_train.T)

    # Apply the same scaling to P_test and T_test using the previously fitted scalers
    p_test = scaler_input.transform(P_test.T)
    t_test = scaler_output.transform(T_test.T)

    # Return the variables
    return scaler_output, scaler_input, p_train, p_test, t_train, t_test, T_train, T_test, P_train, P_test


def Fun_model_results(p_train, t_train, p_test, t_test, T_train, T_test, M, N, cmd, ps_output):
    # 创建并训练SVR模型
    # model = SVR(C=cmd['c'], gamma=cmd['g'], kernel='rbf', epsilon=0.01)
    # model.fit(p_train, t_train)
    # # 对训练集和测试集进行预测
    # t_sim1 = model.predict(p_train)
    # t_sim2 = model.predict(p_test)

    model = svm_train(t_train.ravel(), p_train, f'-s 3 -t 2 -c {cmd["c"]} -g {cmd["g"]} -p 0.01')
    t_sim1, _, _ = svm_predict(t_train, p_train, model)
    t_sim2, _, _ = svm_predict(t_test, p_test, model)
    t_sim1 = np.array(t_sim1)
    t_sim2 = np.array(t_sim2)
    # 反归一化
    scaler = ps_output

    T_sim1 = scaler.inverse_transform(t_sim1.reshape(-1, 1)).flatten()  # 训练集的预测值
    T_sim2 = scaler.inverse_transform(t_sim2.reshape(-1, 1)).flatten()  # 测试集的预测值

    # 计算结果
    rmse1 = np.sqrt(np.sum((T_sim1 - T_train) ** 2) / M)
    rmse2 = np.sqrt(np.sum((T_sim2 - T_test) ** 2) / N)

    R1 = 1 - np.linalg.norm(T_train - T_sim1) ** 2 / np.linalg.norm(T_train - np.mean(T_train)) ** 2
    R2 = 1 - np.linalg.norm(T_test - T_sim2) ** 2 / np.linalg.norm(T_test - np.mean(T_test)) ** 2

    MRE1 = np.mean(np.abs(T_sim1 - np.mean(T_train)) / T_train)
    MRE2 = np.mean(np.abs(T_sim2 - np.mean(T_test)) / T_test)

    return R1, rmse1, MRE1, R2, rmse2, MRE2, model

def myfun(x):
    global p_train, p_test, t_train, t_test, ps_output, ps_input, selectmax
    # 加载数据文件
    all_t_mic = scipy.io.loadmat('D:\web\web_gis\djangoProject\F\lanshu\特征选取\\all_t_mic_4yue.mat')['all_t_mic']
    all_f_mic = scipy.io.loadmat('D:\web\web_gis\djangoProject\F\lanshu\特征选取\\all_f_mic_4yue.mat')['all_f_mic']
    all_a_co = scipy.io.loadmat('D:\web\web_gis\djangoProject\F\lanshu\特征选取\\all_a_co_4yue.mat')['all_a_co']

    # 对特征进行排序
    sorted_indices = np.argsort(x)[::-1]

    indexselect = sorted_indices[:selectmax]

    # 反归一化训练集、测试集
    scaler_output = ps_output
    scaler_input = ps_input

    P_train = scaler_input.inverse_transform(p_train).T
    P_test = scaler_input.inverse_transform(p_test).T
    T_train = scaler_output.inverse_transform(t_train).T
    T_test = scaler_output.inverse_transform(t_test).T

    # 外面也会改变
    P_train = P_train[indexselect, :].T
    P_test = P_test[indexselect, :].T
    T_train = T_train.T
    T_test = T_test.T

    # 定义归一化后的训练集
    p_tr = p_train[:, indexselect]
    t_tr = t_train
    # 定义归一化后的测试集
    p_te = p_test[:, indexselect]
    t_te = t_test

    # 计算选择的各特征与真值的相关性 t_co
    q = []
    for j in range(selectmax):
        p = p_tr[:, j]
        q.append(np.corrcoef(p[:, np.newaxis].T, t_tr.T)[0, 1])  # Pearson correlation
    t_co = np.sum(q) / selectmax

    # 计算选择的特征间自相关性 f_co
    u = np.zeros((selectmax, selectmax))
    for l in range(selectmax - 1):
        for b in range(l + 1, selectmax):
            u[l, b] = np.corrcoef(p_tr[:, l].T, p_tr[:, b].T)[0, 1]
    f_co = np.sum(u) / ((selectmax - 1) * selectmax / 2)

    # 计算选择的特征代表全特征的能力 a_co
    A = []
    for j in range(selectmax):
        A.append(all_a_co[indexselect[j], :])
    b = np.mean(A, axis=1)
    a_co = np.sum(b) / selectmax

    # 计算特征与真值之间的互信息 t_mic
    select_mic = all_t_mic[indexselect, :]
    t_mic = np.sum(select_mic) / selectmax

    # 计算选择的特征之间的互信息 f_mic
    total_mutual_info = 0
    for i in range(selectmax - 1):
        for j in range(i + 1, selectmax):
            feature1 = indexselect[i]
            feature2 = indexselect[j]
            mutual_info_value = all_f_mic[min(feature1, feature2), max(feature1, feature2)]
            total_mutual_info += mutual_info_value
    f_mic = np.sum(np.sum(total_mutual_info)) / (((selectmax - 1) * selectmax) / 2)

    # 计算适应度函数
    fitness_value = (2 * a_co + t_co + t_mic) / (f_co + f_mic)

    return fitness_value, a_co, t_co, f_co, t_mic, f_mic

def initPositions(SearchAgents_no, feature_num, ub, lb):
    global p_train, p_test, t_train, t_test, ps_output, ps_input, selectmax
    BackPositions = np.zeros((SearchAgents_no, feature_num))
    np.random.seed()  # Initialize random seed based on current time
    PositionsF = np.random.rand(SearchAgents_no, feature_num) * (ub - lb) + lb  # Random between lb and ub
    BackPositions = ub + lb - PositionsF  # Back transformation

    # Reverse operation
    PBackPositions = np.zeros_like(BackPositions)
    for i in range(SearchAgents_no):
        for j in range(feature_num):
            if (ub + lb) / 2 < BackPositions[i, j]:
                Lb = (ub + lb) / 2
                Ub = BackPositions[i, j]
                PBackPositions[i, j] = (Ub - Lb) * np.random.rand() + Lb
            else:
                Lb = BackPositions[i, j]
                Ub = (ub + lb) / 2
                PBackPositions[i, j] = (Ub - Lb) * np.random.rand() + Lb

        # Combine populations
        AllPositionsTemp = np.vstack((PositionsF, PBackPositions))
        AllPositions = AllPositionsTemp
        fitness = np.zeros((AllPositionsTemp.shape[0], 1))
        for i in range(AllPositionsTemp.shape[0]):
            y, _, _, _, _, _ = myfun(AllPositionsTemp[i, :])
            fitness[i, 0] = y

        index = np.argsort(fitness[:, 0])[::-1]

        for i in range(2 * SearchAgents_no):
            AllPositions[i, :] = AllPositionsTemp[index[i], :]

        # Select the best fitness-ranked positions as the initial population
        # Step 1: 提取前 `SearchAgents_no` 行
        Positions = AllPositions[:SearchAgents_no, :]

        # Step 2: 对 Positions 的每一行按降序排序
        Positions_des = np.sort(Positions, axis=1)[:, ::-1]  # 按行排序，降序
        pos_sort = np.argsort(Positions, axis=1)[:, ::-1]  # 排序的索引

        # Step 3: 更新 fitness，只取前 `SearchAgents_no` 行
        fitness = fitness[:SearchAgents_no, :]

        # Step 4: 拼接 fitness 和 pos_sort
        fitness_pos = np.hstack((fitness, pos_sort))

        return Positions, fitness_pos


# 替代mapminmax的归一化函数
def mapminmax(data, min_val=0, max_val=1):
    """
    将数据归一化到[min_val, max_val]区间。

    参数：
    data : numpy.ndarray
        输入数据，形状为 (n_samples, n_features)
    min_val : float, 可选，默认 0
        归一化后的最小值
    max_val : float, 可选，默认 1
        归一化后的最大值

    返回：
    normalized_data : numpy.ndarray
        归一化后的数据
    scaler : sklearn.preprocessing.MinMaxScaler
        归一化使用的Scaler对象，包含了最小值和最大值，可以用于应用相同的变换到其他数据
    """
    # 创建 MinMaxScaler 对象并进行拟合
    scaler = MinMaxScaler(feature_range=(min_val, max_val))

    # 将数据进行归一化
    normalized_data = scaler.fit_transform(data.T).T

    return normalized_data, scaler


def mapminmax_apply(data, scaler):
    """
    使用已有的归一化参数将新的数据应用到同样的归一化范围。

    参数：
    data : numpy.ndarray
        新的输入数据，形状为 (n_samples, n_features)
    scaler : sklearn.preprocessing.MinMaxScaler
        归一化时使用的Scaler对象，包含了最小值和最大值

    返回：
    transformed_data : numpy.ndarray
        归一化后的数据
    """
    # 使用已有的 scaler 对数据进行转换
    transformed_data = scaler.transform(data.T).T

    return transformed_data

def update_pos(Positions, SearchAgents_no, feature_num, a, T, Alpha_position, Beta_position, Delta_position, lb, ub):
    """
    更新位置函数，模拟 GWO 算法的更新规则。

    参数：
    - Positions: 当前种群的位置 (SearchAgents_no, feature_num)
    - SearchAgents_no: 搜索代理的数量
    - feature_num: 每个代理的特征数量
    - a: 影响搜索步长的参数
    - T: 最大迭代次数
    - Alpha_position: Alpha 位置
    - Beta_position: Beta 位置
    - Delta_position: Delta 位置
    - lb: 特征的下界
    - ub: 特征的上界

    返回：
    - 更新后的 Positions: 代理的新位置
    """

    for i in range(SearchAgents_no):
        for j in range(feature_num):
            # 第一个随机值
            r1 = np.random.rand()
            r2 = np.random.rand()

            # 计算 A1 和 C1
            A1 = 2 * a * r1 - a
            C1 = 0.5 + (0.5 * np.exp(-j / 500)) + (1.4 * (np.sin(j) / 30))

            # 计算 Alpha 位置的更新
            D_alpha = np.abs(C1 * Alpha_position.flatten()[j] - Positions[i, j])
            X1 = Alpha_position.flatten()[j] - A1 * D_alpha

            # 第二个随机值
            r1 = np.random.rand()
            r2 = np.random.rand()

            # 计算 A2 和 C2
            A2 = 2 * a * r1 - a
            C2 = 1 + (1.4 * (1 - np.exp(-j / 500))) + (1.4 * (np.sin(j) / 30))

            # 计算 Beta 位置的更新
            D_beta = np.abs(C2 * Beta_position.flatten()[j] - Positions[i, j])
            X2 = Beta_position.flatten()[j] - A2 * D_beta

            # 第三个随机值
            r1 = np.random.rand()
            r2 = np.random.rand()

            # 计算 A3 和 C3
            A3 = 2 * a * r1 - a
            C3 = (1 / (1 + np.exp(-0.0001 * j / T))) + ((0.5 - 2.5) * ((j / T) ** 2))

            # 计算 Delta 位置的更新
            D_delta = np.abs(C3 * Delta_position.flatten()[j] - Positions[i, j])
            X3 = Delta_position.flatten()[j] - A3 * D_delta

            # 更新当前位置
            Positions[i, j] = (X1 + X2 + X3) / 3

            # 进行边界修正
            Positions[i, j] = np.clip(Positions[i, j], lb, ub)

    return Positions

def selectByGWOfun(P_train, T_train, P_test, T_test, SearchAgents_no, T, dim, feature_num, lb, ub):
    global p_train, p_test, t_train, t_test, ps_output, ps_input, selectmax
    selectmax = dim  # 设置最大特征数

    # 归一化训练集和测试集
    p_train, ps_input = mapminmax(P_train, 0, 1)
    p_test = mapminmax_apply(P_test, ps_input)

    t_train, ps_output = mapminmax(T_train, 0, 1)
    t_test = mapminmax_apply(T_test, ps_output)

    # 转置数据
    p_train = p_train.T
    p_test = p_test.T
    t_train = t_train.T
    t_test = t_test.T

    # 初始化Alpha, Beta, Delta位置
    Alpha_position = np.zeros((1, feature_num))  # 1 行 feature_num 列
    Beta_position = np.zeros((1, feature_num))  # 1 行 feature_num 列
    Delta_position = np.zeros((1, feature_num))  # 1 行 feature_num 列

    # 初始化最优得分
    Alpha_score = -np.inf
    Beta_score = -np.inf
    Delta_score = -np.inf

    # 初始化位置和适应度
    Positions, fitness_pos = initPositions(SearchAgents_no, feature_num, ub, lb)

    iterations = []
    accuracy = []
    param_a = []
    All_position = []
    All_score = []
    First_score = []
    t = 0

    # 迭代开始
    while t < T:
        for i in range(Positions.shape[0]):
            # 边界修正
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * ~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb

            # 计算适应度
            fitness_value, a_co, t_co, f_co, t_mic, f_mic = myfun(Positions[i, :])
            # 可能缺维
            First_score.append([a_co, t_co, f_co, t_mic, f_mic, fitness_value])

            # 更新最优位置
            if fitness_value > Alpha_score:
                Alpha_score = fitness_value
                Alpha_score_a_co, Alpha_score_t_co = a_co, t_co
                Alpha_score_f_co, Alpha_score_t_mic, Alpha_score_f_mic = f_co, t_mic, f_mic
                Alpha_position = Positions[i, :]

            # 更新Beta和Delta位置
            elif fitness_value < Alpha_score and fitness_value > Beta_score:
                Beta_score = fitness_value
                Beta_position = Positions[i, :]

            elif fitness_value < Alpha_score and fitness_value < Beta_score and fitness_value > Delta_score:
                Delta_score = fitness_value
                Delta_position = Positions[i, :]

        # 计算a参数
        a = 0
        if t <= T / 2:
            a = 1 + (abs(math.cos(math.pi * (t - 1) / (T - 1)))) ** 0.7
            param_a.append(a)
        elif T / 2 < t < T:
            a = 1 - (abs(math.cos(math.pi * (t - 1) / (T - 1)))) ** 0.7
            param_a.append(a)

        # 更新位置
        Positions = update_pos(Positions, SearchAgents_no, feature_num, a, T, Alpha_position, Beta_position,
                               Delta_position, lb, ub)

        # 对每个位置进行排序
        pos_sort2 = np.argsort(Positions, axis=1)[:, ::-1]

        t += 1
        iterations.append(t)
        accuracy.append(Alpha_score)

        print(f'-----------------{t}-th iteration-----------------')

        # 对Alpha_position进行排序并记录位置
        S = np.argsort(Alpha_position)[::-1]

        # 可能缺维度
        All_position.append(S)
        All_score.append(
            [Alpha_score_a_co, Alpha_score_t_co, Alpha_score_f_co, Alpha_score_t_mic, Alpha_score_f_mic, Alpha_score])
    return Alpha_position, iterations, accuracy, param_a, All_position, All_score

def Fun_All_Factors2(A, m0, n0, ck):
    # 提取各个波段
    Ar = A[:, :, 2]  # 第三波段 (红光)
    Ag = A[:, :, 1]  # 第二波段 (绿光)
    Ab = A[:, :, 0]  # 第一波段 (蓝光)
    Are = A[:, :, 3]  # 第四波段 (红边)
    Anir = A[:, :, 4]  # 第五波段 (近红外)
    gray_Anir = exposure.rescale_intensity(Anir, out_range=(0, 1))  # 归一化

    # 初始化各个记录矩阵
    Rstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Gstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Bstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Rmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    Gmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    Bmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    NDVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    EVI2_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    RVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    DVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    RDVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    MSR_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    MCARI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    OSAVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    WDRVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    NIR_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    S1 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S2 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S3 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S4 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S5 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S6 = np.zeros((int(m0 / ck), int(n0 / ck)))

    # 滑动窗口
    with tqdm.tqdm(total=int(m0 / ck), desc="数据保存") as pbar:
        for i in range(int(m0 / ck)):
            for j in range(int(n0 / ck)):
                # 计算窗口位置
                h = int(ck - (ck - 1) / 2 + i * ck)
                l = int(ck - (ck - 1) / 2 + j * ck)
                w = int(h - (ck - 1) / 2) - 1
                e = int(l - (ck - 1) / 2) - 1

                # 提取窗口
                window_Ar = Ar[w:w + ck, e:e + ck]
                window_Ag = Ag[w:w + ck, e:e + ck]
                window_Ab = Ab[w:w + ck, e:e + ck]
                window_Are = Are[w:w + ck, e:e + ck]
                window_Anir = Anir[w:w + ck, e:e + ck]
                window_gray_Anir = gray_Anir[w:w + ck, e:e + ck]

                # 计算颜色矩阵
                R = np.mean(window_Ar)
                G = np.mean(window_Ag)
                Rstd[i, j] = np.std(window_Ar)
                Gstd[i, j] = np.std(window_Ag)
                Bstd[i, j] = np.std(window_Ab)
                Rmean[i, j] = R
                Gmean[i, j] = G
                Bmean[i, j] = np.mean(window_Ab)

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

                NDVI_record[i, j] = NDVI
                EVI2_record[i, j] = EVI2
                RVI_record[i, j] = RVI
                DVI_record[i, j] = DVI
                RDVI_record[i, j] = RDVI
                MSR_record[i, j] = MSR
                MCARI_record[i, j] = MCARI
                OSAVI_record[i, j] = OSAVI
                WDRVI_record[i, j] = WDRVI
                NIR_record[i, j] = NIR

                image_uint8 = (window_gray_Anir * 63).astype(np.uint8)
                # 纹理指数计算
                glcms1 = graycomatrix(image_uint8, levels=64, distances=[1],
                                      angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
                # np.transpose(glcms1, (0, 1, 3, 2))
                # stats = graycoprops(glcms1, prop=['contrast', 'correlation', 'energy', 'homogeneity'])
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
                S1[i, j] = s1
                S2[i, j] = s2
                S3[i, j] = s3
                S4[i, j] = s4
                S5[i, j] = s5
                S6[i, j] = s6
            pbar.update(1)
    # 将记录矩阵展平并拼接
    NDVI_record = NDVI_record.flatten()
    EVI2_record = EVI2_record.flatten()
    RVI_record = RVI_record.flatten()
    DVI_record = DVI_record.flatten()
    RDVI_record = RDVI_record.flatten()
    MSR_record = MSR_record.flatten()
    MCARI_record = MCARI_record.flatten()
    OSAVI_record = OSAVI_record.flatten()
    WDRVI_record = WDRVI_record.flatten()
    NIR_record = NIR_record.flatten()
    S1 = S1.flatten()
    S2 = S2.flatten()
    S3 = S3.flatten()
    S4 = S4.flatten()
    S5 = S5.flatten()
    S6 = S6.flatten()
    Rmean = Rmean.flatten()
    Gmean = Gmean.flatten()
    Bmean = Bmean.flatten()
    Rstd = Rstd.flatten()
    Gstd = Gstd.flatten()
    Bstd = Bstd.flatten()

    # 拼接所有因子
    outmat = np.column_stack((NDVI_record, EVI2_record, RVI_record, DVI_record, RDVI_record, MSR_record,
                              MCARI_record, OSAVI_record, WDRVI_record, NIR_record, S1, S2, S3, S4,
                              S5, S6, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd))

    return outmat


if __name__ == "__main__":
    # column_to_row_major(np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9], [10, 11, 12]]))
    # myfun(2,2,2,2,2,2,2,2)
    # R = np.array([
    #     [10, 20, 30, 40, 50, 60],
    #     [15, 25, 35, 45, 55, 65],
    #     [20, 30, 40, 50, 60, 70],
    #     [25, 35, 45, 55, 65, 75]
    # ], dtype=np.float64)
    # R /= R.sum()
    # R_ad = exposure.equalize_adapthist(R)
    # R_ad_2 = exposure.equalize_adapthist(R.T).T

    pass
