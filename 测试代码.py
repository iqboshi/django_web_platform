import math
total_steps = 100
step = 10
ck = 25
m = 150
m0 = m - m % ck
for i in range(int(m0 / ck)):
    step += math.floor(90 / int(m0 / ck))
    progress = step / total_steps
    print(progress)
# redis_instance.set(f"task_progress_{task_id}", progress * 100)