import random
import string
import tempfile
import zipfile
import redis
from django.http import JsonResponse, Http404
from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.files.storage import default_storage
from djangoApp.models import PositionPoint, Image, Task
from django.shortcuts import get_object_or_404
import os
from .tasks import lanshu_soil_remove, lanshu_feature_select, task_picture_synthesis
from django.conf import settings
# Create your views here.
@csrf_exempt
def API_SEND_POINTS(request):
    data = json.loads(request.body)['data']
    for item in data:
        lng = item.get('lng')
        lat = item.get('lat')
        # print(lng, lat)
        if lng is not None and lat is not None:
            PositionPoint.objects.create(longitude=lng, latitude=lat)
    return HttpResponse("API_SEND_POINTS")

@csrf_exempt
def API_GET_POINTS(request):
    # points =PositionPoint.objects.all().values('latitude', 'longitude', 'created_at', 'description', 'pk')
    points = PositionPoint.objects.prefetch_related('images').all()
    result = []

    for point in points:
        # 通过模型实例直接访问 images 属性
        image_urls = [image.image_file.url for image in point.images.all()]
        # 手动构造结果字典，确保所有需要的字段都被包括
        point_data = {
            'latitude': point.latitude,
            'longitude': point.longitude,
            'created_at': point.created_at.strftime('%Y-%m-%d %H:%M:%S'),  # 格式化日期
            'description': point.description,
            'pk': point.pk,
            'images': image_urls,
        }
        result.append(point_data)
        # print(point_data)
    # 使用 JsonResponse 发送数据
    return JsonResponse(result, safe=False)


@csrf_exempt
def API_DELETE_POINTS(request):
    pk = request.GET.get('id')
    PositionPoint.objects.filter(pk=pk).delete()
    return HttpResponse("API_DELETE_POINTS")

@csrf_exempt
def API_EDIT_POINTS(request):
    pk = int(request.POST.get('id'))
    describe = request.POST.get('describe')
    # 获取主键为 pk 的 PositionPoint 对象
    position_point = get_object_or_404(PositionPoint, pk=pk)
    position_point.description = describe
    position_point.save()
    uploaded_files = request.FILES.getlist('files')
    # print(uploaded_files)
    for uploaded_file in uploaded_files:
        # 创建一个 Image 实例并保存
        image_instance = Image.objects.create(image_file=uploaded_file)
        # 关联 Image 实例到 PositionPoint 对象
        position_point.images.add(image_instance)
    return HttpResponse("API_EDIT_POINTS")

# 初始化 Redis 连接
redis_instance = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)

@csrf_exempt
def API_SOIL_REMOVE(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    uploaded_file = request.FILES.get('file', None)
    if not uploaded_file:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    # 将上传的文件保存到默认存储路径（通常是MEDIA_ROOT下的uploads目录）
    file_path = default_storage.save(os.path.join('uploads', uploaded_file.name), uploaded_file)
    # 调用Celery任务来处理这个文件
    task = lanshu_soil_remove.delay()
    # print(task)
    # 返回任务 ID，以便前端轮询获取任务的状态
    return JsonResponse({'task_id': task.id}, status=202)

@csrf_exempt
def API_FEATURE_SELECT(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    uploaded_file = request.FILES.get('file', None)
    if not uploaded_file:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    # 将上传的文件保存到默认存储路径（通常是MEDIA_ROOT下的uploads目录）
    file_path = default_storage.save(os.path.join('uploads', uploaded_file.name), uploaded_file)
    # 调用Celery任务来处理这个文件
    task = lanshu_feature_select.delay()
    # print(task)
    # 返回任务 ID，以便前端轮询获取任务的状态
    return JsonResponse({'task_id': task.id}, status=202)


@csrf_exempt
def API_IMAGE_SYNTHESIS(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    uploaded_files = request.FILES.getlist('files')
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    saved_files = []
    # print(uploaded_files)
    # 遍历上传的文件并保存
    for file in uploaded_files:
        # 获取原文件名和文件扩展名
        filename, file_extension = os.path.splitext(file.name)

        # 生成一个随机字符串作为文件名的一部分
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        # 新的文件名：原文件名 + 随机数 + 扩展名
        new_filename = f"{filename}_{random_str}{file_extension}"

        # 完整的文件路径
        file_path = os.path.join(upload_dir, new_filename)

        # 保存文件到指定目录
        with open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        saved_files.append(file_path)
    # print(saved_files)
    task = task_picture_synthesis.delay(files=saved_files)
    return JsonResponse({'task_id': task.id}, status=202)

@csrf_exempt
def get_task_progress(request, task_id):
    try:
        # 从 Redis 中获取任务进度
        progress = redis_instance.get(f"task_progress_{task_id}")
        print(progress)
        if progress is None:
            return JsonResponse({'state': 'PENDING', 'progress': 0}, status=200)
        if float(progress) < 100:
            return JsonResponse({'state': 'PROGRESS', 'progress': float(progress)}, status=200)
        else:
            return JsonResponse({'state': 'SUCCESS', 'progress': 100.0}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def download_file(request, file_name):
    # 构造文件路径
    file_path = os.path.join(settings.MEDIA_ROOT, 'task_results', file_name)

    # 检查文件是否存在
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/force-download")
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            return response

    raise Http404("File does not exist")

@csrf_exempt
def API_GET_TASKS(request):
    result = []
    tasks = Task.objects.all()
    for task in tasks:
        result.append({
            'pk': task.pk,
            'task_id': task.task_id,
            'task_type': task.task_type,
            'results': task.results,
            'created_at': task.created_at.strftime('%Y-%m-%d %H:%M:%S'),  # 格式化日期
        })
    # 使用 JsonResponse 发送数据
    return JsonResponse(result, safe=False)

@csrf_exempt
def API_TASKS_RESULTS_DOWNLOAD(request):
    try:
        # 获取指定任务对象
        task = Task.objects.get(pk=request.GET.get('id'))

        # 检查是否有结果文件
        if not task.results or len(task.results) == 0:
            return HttpResponse("No results available for this task.", status=404)

        # 创建临时 ZIP 文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_file_path = temp_zip.name

        # 使用 ZipFile 创建压缩包
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            # 遍历结果文件路径，将每个文件添加到 ZIP 中
            for file_name in task.results:
                file_path = os.path.join(settings.MEDIA_ROOT, 'task_results', file_name)
                # print(file_path)
                if os.path.exists(file_path):
                    # 使用 arcname 参数确保文件在 ZIP 中的名称保持简单
                    zipf.write(file_path, arcname=os.path.basename(file_path))
                else:
                    return HttpResponse(f"File {file_name} does not exist.", status=404)

        # 返回 ZIP 文件作为下载响应
        with open(zip_file_path, 'rb') as zip_file:
            response = HttpResponse(zip_file.read(), content_type='application/zip')
            response['Content-Disposition'] = f'attachment; filename="task_results_{task.pk}.zip"'
            response['Content-Length'] = os.path.getsize(zip_file_path)
            return response

    except Task.DoesNotExist:
        return HttpResponse("Task does not exist.", status=404)

    except Exception as e:
        return HttpResponse(f"An error occurred: {str(e)}", status=500)

    finally:
        # 确保临时文件被删除
        if 'zip_file_path' in locals() and os.path.exists(zip_file_path):
            os.remove(zip_file_path)

@csrf_exempt
def API_DELETE_TASKS(request):
    pk = request.GET.get('id')
    Task.objects.filter(pk=pk).delete()
    return HttpResponse("API_DELETE_TASKS")
