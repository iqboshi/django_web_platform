import numpy as np
import hdf5storage
from utils.funcs import Fun_init, Fun_model_results, column_to_row_major, selectByGWOfun
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import redis
from django.conf import settings
import os
from libsvm.svmutil import *
# 设置字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 初始化 Redis 连接

results = []
redis_instance = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)
def plot_convergence(iterations, accuracy, param_a, task_id):

    # 绘制 accuracy 随迭代次数变化的图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy)
    plt.xlabel('Count of iterations', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title('Sanitized GWO-SVM parameter optimization (SGWO_SVM)', fontsize=16)
    plt.grid(True)  # 添加网格
    # plt.show()
    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    file_name = f'SGWO_SVM_accuracy_{task_id}.png'
    file_path = os.path.join(task_results_folder, file_name)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    results.append(file_name)
    # 绘制 收敛因子 a 随迭代次数变化的图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, param_a)
    plt.xlabel('迭代次数', fontsize=20)
    plt.ylabel('收敛因子', fontsize=20)
    plt.title('收敛因子a变化图', fontsize=16)
    plt.grid(True)  # 添加网格
    # plt.show()
    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    file_name = f'SGWO_SVM_paramA_{task_id}.png'
    file_path = os.path.join(task_results_folder, file_name)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    results.append(file_name)
def feature_select(task_id):

    # 加载.mat文件
    A_data = hdf5storage.loadmat('/F/lanshu/A_matlab.mat')
    A = A_data['A_matlab']  # 获取A_matlab变量
    A = np.array(A)
    # column_to_row_major(A)

    factor = ["NDVI", "EVI2", "RVI", "DVI", "RDVI", "MSR", "MCARI", "OSAVI", "WDRVI", "NIR", "对比度", "相关性", "熵",
              "平稳度", "能量", "同质性", "Rmean", "Gmean", "Bmean", "Rstd", "Gstd", "Bstd", "Mask_rate"]
    factor = np.array(factor)

    # p_train, p_test, t_train, t_test, ps_output, ps_input, selectmax = None,None,None,None,None,None,None

    # 设置随机种子
    np.random.seed(2207)

    # 定义总样本数
    total_num = 88

    # 生成随机排列
    temp = np.random.permutation(total_num)

    # 定义训练集和测试集的样本数
    M = 71
    N = 17

    # 使用切片和生成的随机索引来初始化数据
    T_train, T_test, P_train, P_test = Fun_init(A, temp[:M], [i for i in range(23)], temp[M:])[-4:]

    SearchAgents_no = 40
    T = 20
    dim = 13
    feature_num = 23
    lb = 0
    ub = 1
    redis_instance.set(f"task_progress_{task_id}", 5)
    Alpha_position, iterations, accuracy, param_a, All_position, All_score = selectByGWOfun(P_train, T_train, P_test, T_test, SearchAgents_no, T, dim, feature_num, lb, ub)
    redis_instance.set(f"task_progress_{task_id}", 75)
    # 排序 Alpha_position，获取排序后的值和索引（降序排序）
    v_sorted = np.sort(Alpha_position)[::-1]  # 降序排序
    S = np.argsort(Alpha_position)[::-1]  # 获取排序后的索引，降序排序
    # 选择前 dim 个特征
    select_indices = S[:dim]
    indexselect_name = factor[select_indices]
    plot_convergence(iterations, accuracy, param_a, task_id=task_id)

    Y = select_indices

    ps_output, ps_input, p_train, p_test, t_train, t_test, T_train, T_test, _, _ = Fun_init(A, temp[:71], Y, temp[71:])
    redis_instance.set(f"task_progress_{task_id}", 85)

    # 加载 FD2 数据
    mat_data = hdf5storage.loadmat('D:/新建文件夹/蓝姝-代码数据汇总/算法/代码/FD2_data.mat')
    P_valFD2 = np.array(mat_data['FD2_data'])
    P_valFD2 = P_valFD2[:, Y]
    # 生成随机的 t_valFD2 数据
    t_valFD2 = np.random.randint(1, 7, 40128)

    # 使用 MinMaxScaler 进行数据归一化（假设 ps_input 是 scaler 的相关参数）

    p_valFD2 = ps_input.transform(P_valFD2)  # 归一化时需要转置

    # 设置 SVM 参数
    c = 5.43
    g = 0.143
    t = 2
    s = 3
    p = 0.01
    # cmd = f" -t 2 -c {c} -g {g} -s 3 -p 0.01"
    cmd = {
        'c': c,
        'g': g,
        't': t,
        's': s,
        'p': p,
    }
    R1, rmse1, MRE1, R2, rmse2, MRE2, model = Fun_model_results(p_train, t_train, p_test, t_test, T_train, T_test, M, N, cmd, ps_output)
    # FD2 数据预测部分
    # t_valFD2 是预测标签，p_valFD2 是输入特征
    # 使用 model 对 t_valFD2 和 p_valFD2 进行预测
    t_simFD2, _, _ = svm_predict(t_valFD2, p_valFD2, model)
    t_simFD2 = np.array(t_simFD2)

    # 使用 MinMaxScaler 进行反归一化操作
    scaler_output = ps_output
    # 注意维度，可能出错
    T_simFD2 = scaler_output.inverse_transform(t_simFD2[:, np.newaxis])
    # 将预测结果转换为期望的形状
    FD2_predict = np.reshape(T_simFD2, (418, 96), order='F')

    # 归一化 FD2_predict 数据
    scaler_fd2 = MinMaxScaler(feature_range=(0, 1))
    FD2_predict_test = scaler_fd2.fit_transform(FD2_predict)
    FD2_predict_test = np.rot90(FD2_predict_test)

    # FD2_predict_mat = hdf5storage.loadmat("FD2_predict.mat")  # 加载 .mat 文件
    # FD2_predict = FD2_predict_mat['FD2_predict']  # 提取 FD2_predict 数据
    # FD2_predict_test = np.rot90(FD2_predict)  # 顺时针旋转矩阵

    # 绘制图像
    plt.figure(4)
    # 自定义颜色映射
    customColors = np.array([
        [1, 1, 0],  # yellow
        [1, 1, 1],  # white
        [0, 0.5, 0],  # green
        [0, 1, 1]  # cyan
    ])

    # 创建自定义的颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', customColors)

    # 绘制图像
    im = plt.imshow(FD2_predict_test, cmap=custom_cmap, interpolation='nearest')

    # 设置坐标轴
    plt.axis('equal')  # 保持坐标轴比例相同
    plt.axis('off')  # 隐藏坐标轴

    # 添加颜色条
    cbar = plt.colorbar(im, orientation='horizontal')  # 水平颜色条

    # 设置颜色条标签
    cbar.set_label('tiller number (tillers/plant)', fontsize=14, fontweight='bold')

    # 设置字体
    plt.gca().tick_params(labelsize=14)
    plt.gca().set_facecolor('white')  # 设置背景颜色为白色

    task_results_folder = os.path.join(settings.MEDIA_ROOT, 'task_results')
    file_name = f'FD2_{task_id}.png'
    file_path = os.path.join(task_results_folder, file_name)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    results.append(file_name)
    redis_instance.set(f"task_progress_{task_id}", 100)
    return results