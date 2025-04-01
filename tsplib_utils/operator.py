import random
import math
import numpy as np

##############################
#########  MUTATION  #########
##############################

def naive_swap(tour: list[int], distance_matrix=None):
    """随机交换两个城市的位置"""
    tour = tour[:]
    i, j = random.choices(range(len(tour)), k=2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour


def chunk_swap(tour: list[int], distance_matrix=None):
    """随机交换两个子序列的位置"""
    tour = tour[:]
    if len(tour) < 4:
        return tour  # 不足以交换
    i, j, p, q = random.sample(range(len(tour)), k=4)
    i, j, p, q = sorted((i, j, p, q))
    tour = tour[:i] + tour[p:q + 1] + tour[j + 1:p] + tour[i:j + 1] + tour[q + 1:]
    return tour


def naive_insert(tour: list[int], distance_matrix=None):
    """随机选择一个城市并将其插入到另一个位置"""
    tour = tour[:]
    if len(tour) < 2:
        return tour
    i, j = random.choices(range(len(tour)), k=2)
    i, j = sorted((i, j))
    city = tour.pop(j)
    tour.insert(i + 1, city)
    return tour


def greedy_insert(tour: list[int], distance_matrix: np.ndarray):
    """贪心插入操作，尝试以最小增量代价插入城市"""
    tour = tour[:]
    dimension = len(distance_matrix)
    times = int(max(random.gauss(0.018, 0.01), 2 / dimension) * dimension)
    if random.random() < 0.1:
        times = max(dimension // random.randint(10, 30), 1)

    if random.random() < 0.5:
        conductor = [tour.pop(random.randint(1, len(tour) - 2)) for _ in range(times)]
    else:
        pivot = random.randint(1, dimension - times - 1)
        conductor = [tour.pop(pivot) for _ in range(times)]

    for item in conductor:
        vertex = item
        tour.append(tour[0])
        best_gain, best_idx = math.inf, -1
        for j in range(1, len(tour)):
            # 转成 0 基索引
            a = tour[j - 1] - 1
            b = vertex - 1
            c = tour[j] - 1

            # 计算增量
            gain = distance_matrix[a][b] + distance_matrix[b][c] - distance_matrix[a][c]
            if gain < best_gain and random.random():
                best_gain = gain
                best_idx = j
        del tour[-1]
        tour.insert(best_idx, vertex)

    return tour


def chunk_insert(tour: list[int], distance_matrix=None):
    """随机选择一个子序列并将其插入到另一个位置"""
    tour = tour[:]
    if len(tour) < 3:
        return tour  # 不足以插入
    i, j, k = random.sample(range(len(tour)), k=3)
    # 确保 i < j < k
    i, j, k = sorted((i, j, k))
    # 提取子序列
    sub_seq = tour[i:j + 1]
    del tour[i:j + 1]
    # 插入子序列
    insert_pos = k - (j - i + 1)  # 调整插入位置
    tour = tour[:insert_pos] + sub_seq + tour[insert_pos:]
    return tour


def naive_reverse(tour: list[int], distance_matrix=None):
    """随机选择两个位置并反转中间的子序列"""
    tour = tour[:]
    if len(tour) < 3:
        return tour  # 不足以反转
    i, j = random.choices(range(1, len(tour)), k=2)
    i, j = sorted((i, j))
    tour[i:j + 1] = tour[i:j + 1][::-1]
    return tour


def opt_swap_2(tour: list[int], distance_matrix=None):
    """2-opt 交换，反转两个非重叠的边之间的路径"""
    tour = tour[:]
    if len(tour) < 4:
        return tour  # 不足以交换
    i = random.randint(0, len(tour) - 4)
    j = random.randint(i + 2, len(tour) - 2)
    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
    return tour


def opt_swap_3(tour: list[int], distance_matrix=None):
    """3-opt 交换，涉及三个边的路径交换"""
    tour = tour[:]
    if len(tour) < 6:
        return tour  # 不足以交换
    i = random.randint(0, len(tour) - 6)
    j = random.randint(i + 2, len(tour) - 4)
    k = random.randint(j + 2, len(tour) - 2)
    
    odds = random.randint(1, 4)
    if odds == 1:
        tour = tour[:i + 1] + tour[k:j:-1] + tour[i + 1:j + 1] + tour[k + 1:]
    elif odds == 2:
        tour = tour[:i + 1] + tour[j + 1:k + 1] + tour[i + 1:j + 1] + tour[k + 1:]
    elif odds == 3:
        tour = tour[:i + 1] + tour[j + 1:k + 1] + tour[j:i:-1] + tour[k + 1:]
    else:
        tour = tour[:i + 1] + tour[j:i:-1] + tour[k:j:-1] + tour[k + 1:]
    return tour

##############################
######### CROSSOVER #########
##############################

def ox(p1: list[int], p2: list[int]):
    """order crossover (OX)"""

    # Choose an arbitrary part from the parent
    i, j = random.sample(range(1, len(p1) - 1), k=2)
    i, j = sorted((i, j))

    def breed(mother, father):
        nonlocal i, j
        # Copy this part to the first child
        offspring = [0 for _ in range(len(mother))]
        offspring[i:j] = mother[i:j]

        # Copy the numbers that are not in the first part, to the first child:
        existed = set(offspring[i:j])
        writer = 0
        for p in range(len(father)):
            if writer >= len(offspring):
                break

            while offspring[writer] != 0:
                writer += 1

            # using the order of the second parent
            if father[p] not in existed:
                offspring[writer] = father[p]
                existed.add(father[p])
                writer += 1

        return offspring

    return breed(p1, p2), breed(p2, p1)


def pmx(p1: list[int], p2: list[int]):
    """partially mapped crossover (PMX)"""
    i, j = random.sample(range(1, len(p1) - 2), k=2)
    i, j = sorted((i, j))

    # naive crossover
    o1 = p1[:i] + p2[i:j + 1] + p1[j + 1:]
    o2 = p2[:i] + p1[i:j + 1] + p2[j + 1:]

    # mapping
    mapping, circles = {}, []
    for idx in range(i, j + 1):
        if p1[idx] not in mapping and p2[idx] not in mapping:
            circles.append([p1[idx], p2[idx]])
            mapping[p1[idx]] = len(circles) - 1
            mapping[p2[idx]] = len(circles) - 1
        elif p1[idx] not in mapping:
            mapping[p1[idx]] = mapping[p2[idx]]
            circles[mapping[p2[idx]]].append(p1[idx])
        elif p2[idx] not in mapping:
            mapping[p2[idx]] = mapping[p1[idx]]
            circles[mapping[p1[idx]]].append(p2[idx])
        else:
            circles[mapping[p1[idx]]].extend(circles[mapping[p2[idx]]])
            for e in circles[mapping[p2[idx]]]:
                mapping[e] = mapping[p1[idx]]

    # resolve conflict
    used = set(o1[i:j + 1])
    for idx in range(len(o1)):
        if (idx < i or idx > j) and o1[idx] in used:
            for n in circles[mapping[o1[idx]]]:
                if n not in used and n != o1[idx]:
                    o1[idx] = n
                    used.add(n)
                    break

    used = set(o2[i:j + 1])
    for idx in range(len(o2)):
        if (idx < i or idx > j) and o2[idx] in used:
            for n in circles[mapping[o2[idx]]]:
                if n not in used and n != o1[idx]:
                    o2[idx] = n
                    used.add(n)
                    break

    return o1, o2
