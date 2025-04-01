import numpy as np

def calculate_distance(tour, distance_matrix):
    """计算给定tour的总路程"""
    # 调整为0基索引
    return sum(distance_matrix[tour[i]-1, tour[(i+1)%len(tour)]-1] for i in range(len(tour)))

def linear_rank_probs(n):
    """
    线性排名选择概率，返回规模为n的概率列表。
    rank=0 -> 最优, rank=n-1 -> 最差
    p(rank_i) = 2*(n-rank_i)/(n*(n+1))
    """
    rank_probs = []
    for rank_i in range(n):
        p = 2.0 * (n - rank_i) / (n*(n+1))
        rank_probs.append(p)
    # 归一化
    total_p = sum(rank_probs)
    rank_probs = [p / total_p for p in rank_probs]
    return rank_probs

def rank_selection(candidates, distances, keep_size, avoid_duplicate=True):
    """
    基于'线性排名'从 candidates 中选 keep_size 个。
    candidates: list of tours
    distances : list of float, 与 candidates 一一对应
    keep_size : 需要保留或选出的数量
    avoid_duplicate : 是否避免重复解
    """
    # 按照距离由小到大排序
    sorted_indices = sorted(range(len(candidates)), key=lambda i: distances[i])
    n = len(candidates)

    # 计算排名选择概率
    rank_probs = linear_rank_probs(n)

    next_gen = []
    while len(next_gen) < keep_size:
        idx_choice = np.random.choice(n, p=rank_probs)
        chosen_tour = candidates[sorted_indices[idx_choice]]
        if (not avoid_duplicate) or (chosen_tour not in next_gen):
            next_gen.append(chosen_tour)
    return next_gen


