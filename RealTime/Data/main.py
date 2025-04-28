import numpy as np
import random

def read_and_process_groups(filename):
    all_groups = []
    multiple_groups = []
    

    with open(filename, 'r') as file:
        for line in file:
            # 去除空白字符和注释行
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # 将逗号分隔的字符串转换为整数列表
            group = [int(num) for num in line.split(',')]
            all_groups.append(group)
            
            # 如果是单个元素的组，添加到 single_groups
            if len(group) > 1:
                multiple_groups.append(group)
    
    return all_groups, multiple_groups
def find_array_place(point, array):
    for i in range(len(array)):
        if point in array[i]:
            return i
    return None

# 测试代码
if __name__ == "__main__":
    all_arrays, multiple_arrays = read_and_process_groups('groups.txt')
    for k in [1,3,5]:
        for a in [3,6,9]:
            start = random.randint(0, 72)
            end = random.randint(0, 72)
            while start == end:
                end = random.randint(0, 72)

            usage_count = np.zeros(73)
            usage_count[start] = 1
            usage_count[end] = 1
            init = []
            new = []
            for _ in range (k*a//3):
                start_window = random.uniform(0, 6)
                end_window = random.uniform(0, 6)
                while abs(start_window - end_window) < 1:
                    start_window = random.uniform(0, 6)
                    end_window = random.uniform(0, 6)
                if start_window > end_window:
                    start_window, end_window = end_window, start_window
                temp_pickup = random.randint(0, 72)
                while usage_count[temp_pickup] >= 3 or temp_pickup == start or temp_pickup == end:
                    temp_pickup = random.randint(0, 72)
                pickup = temp_pickup + 73 * usage_count[temp_pickup]
                usage_count[temp_pickup] += 1
                temp_dropoff = random.randint(0, 72)
                
                while usage_count[temp_dropoff] >= 3 or temp_dropoff == start or temp_dropoff == end or temp_dropoff == temp_pickup:
                    temp_dropoff = random.randint(0, 72)
                dropoff = temp_dropoff + 73 * usage_count[temp_dropoff]
                usage_count[temp_dropoff] += 1
                load = random.randint(10, 30)
                init.append((pickup, dropoff, start_window, end_window, load))
            for _ in range (k*a//3*2):
                start_window = random.uniform(0, 6)
                end_window = random.uniform(0, 6)
                while abs(start_window - end_window) < 1:
                    start_window = random.uniform(0, 6)
                    end_window = random.uniform(0, 6)
                if start_window > end_window:
                    start_window, end_window = end_window, start_window
                flag = True
                while flag == True:
                    temp_pickup = random.choice(multiple_arrays)
                    temp_dropoff = random.choice(multiple_arrays)
                    flag = False
                    if temp_pickup[0] == temp_dropoff[0]:
                        flag = True
                    for sample in temp_pickup:
                        if usage_count[sample] >= 3 or sample == start or sample == end:
                            flag = True
                            break
                    for sample in temp_dropoff:
                        if usage_count[sample] >= 3 or sample == start or sample == end:
                            flag = True
                            break
                pickup = []
                dropoff = []
                for sample in temp_pickup:
                    real_sample = sample + 73 * usage_count[sample]
                    usage_count[sample] += 1
                    pickup.append(real_sample)
                for sample in temp_dropoff:
                    real_sample = sample + 73 * usage_count[sample]
                    usage_count[sample] += 1
                    dropoff.append(real_sample)
                load = random.randint(10, 30)
                new.append((pickup,dropoff, start_window, end_window, load))
                    
            for b in [0,0.5,1]:
                output_file = f'demand/output_{k}_{a}_{b}_init.txt'
                with open(output_file, 'w') as f:
                    f.write(f"{start} {end} {k*a//3}\n\n")
                    for pickup, dropoff, start_window, end_window, load in init:
                        f.write(f"{load} {start_window:.2f} {end_window:.2f}\n")
                        f.write(f"{int(pickup)}\n\n")
                        f.write(f"{int(dropoff)}\n\n")
                output_file = f'demand/output_{k}_{a}_{b}_new.txt'
                with open(output_file, 'w') as f:
                    f.write(f"{2*k*a//3}\n\n")
                    for i in range(int((1-b)*len(new))):
                        pickup, dropoff, start_window, end_window,load = new[i]
                        f.write(f"{load} {start_window:.2f} {end_window:.2f}\n")
                        f.write(f"{int(pickup[0])}\n\n")
                        f.write(f"{int(dropoff[0])}\n\n")
                    for i in range(int((1-b)*len(new)), len(new)):
                        pickup, dropoff, start_window, end_window,load= new[i]
                        f.write(f"{load} {start_window:.2f} {end_window:.2f}\n")
                        for p in pickup:
                            f.write(f"{int(p)} ")
                        f.write("\n\n")
                        for d in dropoff:
                            f.write(f"{int(d)} ")
                        f.write("\n\n")
            
                    
            
