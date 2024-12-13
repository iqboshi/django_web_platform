# <app_name>/tasks.py
from celery import shared_task
from F.lanshu.Quzhou_2024_Soil_remove import soil_remove
from F.lanshu.Feature_Selection import feature_select
from F.zmq.Picture_Synthesis import picture_synthesis
from F.zmq.Collected_Data_Processor import collected_data_processor
from djangoApp.models import Task
@shared_task(bind=True)
def lanshu_soil_remove(self, file_path='D:/新建文件夹/蓝姝-代码数据汇总/算法/数据/原始数据/Original Data/tif/fd1_5_T4.tif'):
    task_id = self.request.id
    # 调用 soil_remove 函数并传递 task_id
    result = soil_remove(task_id=task_id, path=file_path)
    task = Task.objects.create(
        task_id=task_id,
        task_type='type1',
        results=[result]  # 结果字段中存储文件路径
    )
    return {'status': True}

@shared_task(bind=True)
def task_picture_synthesis(self, files):
    task_id = self.request.id
    # print(task_id)
    results = picture_synthesis(task_id=task_id, files=files)
    # print(results)
    task = Task.objects.create(
        task_id=task_id,
        task_type='原始图像合成',
        results=results  # 结果字段中存储文件路径
    )
    return {'status': True}

@shared_task(bind=True)
def task_collected_data_processor(self, files):
    task_id = self.request.id
    results = collected_data_processor(task_id=task_id, files=files, appendmat=True if len(files) > 2 else False)
    task = Task.objects.create(
        task_id=task_id,
        task_type='采集数据处理',
        results=results  # 结果字段中存储文件路径
    )
    return {'status': True}

@shared_task(bind=True)
def lanshu_feature_select(self):
    task_id = self.request.id
    # 调用 soil_remove 函数并传递 task_id
    results = feature_select(task_id=task_id)
    task = Task.objects.create(
        task_id=task_id,
        task_type='特征选取',
        results=results  # 结果字段中存储文件路径
    )
    return {'status': True}

@shared_task
def multiply(x, y):
    return x * y