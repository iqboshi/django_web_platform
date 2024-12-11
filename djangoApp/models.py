from django.db import models

# Create your models here.
# Image 模型，用于存储图片信息
class Image(models.Model):
    image_file = models.ImageField(upload_to='images/')  # 图片字段
    uploaded_at = models.DateTimeField(auto_now_add=True)  # 上传时间

    def __str__(self):
        return f"Image ({self.id}) - {self.image_file.name}"


# PositionPoint 表，存放位置相关信息
class PositionPoint(models.Model):
    latitude = models.DecimalField(max_digits=9, decimal_places=6)  # 纬度
    longitude = models.DecimalField(max_digits=9, decimal_places=6)  # 经度
    description = models.TextField(blank=True, null=True)  # 位置点的描述，可以为空
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间
    images = models.ManyToManyField(Image, related_name='points', blank=True)  # 多张图片关联字段

    def __str__(self):
        return f"PositionPoint at ({self.latitude}, {self.longitude})"


class Task(models.Model):
    TASK_TYPES = (
        ('土壤去除', 'Type 1'),
        ('特征选取', 'Type 2'),
        ('type3', 'Type 3'),
    )

    task_id = models.CharField(max_length=255, unique=True)
    task_type = models.CharField(max_length=50, choices=TASK_TYPES)
    results = models.JSONField(blank=True, null=True)  # 存储多个文件路径
    created_at = models.DateTimeField(auto_now_add=True)  # 上传时间

    def __str__(self):
        return f'Task {self.task_id} ({self.get_task_type_display()})'
