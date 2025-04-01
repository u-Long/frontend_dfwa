import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from tsplib_utils.time import timeit
from tsplib_utils.operator import *
from tsplib_utils.readTSP import read_tsp_file
from tsplib_utils.tools import calculate_distance

class DFWAlgorithm:
    def __init__(self, 
                 filename: Optional[str] = None,
                 distance_matrix: Optional[np.ndarray] = None,
                 init_fireworks: List[List[int]] = None,
                 tag: str = 'DFireworkAlgorithm',
                 verbose: bool = False,
                 boost: bool = False,
                 epoch: int = 5,
                 early_stop: int = 5000,
                 size: int = 50,
                 alpha: float = 1,
                 alpha_: float = 0.9,
                 nspk_factor: int = 200,
                 amp: int = 30):
        
        # 参数校验
        if filename is None and distance_matrix is None:
            raise ValueError("必须提供 filename 或 distance_matrix 至少一个参数")
            
        # 初始化核心参数
        self.tag = tag
        self.verbose = verbose
        self.boost = boost
        self.epoch = epoch
        self.early_stop = early_stop
        self.size = size
        self.alpha = alpha
        self.alpha_ = alpha_
        self.nspk_factor = nspk_factor
        self.amp = amp
        self.jumpout = 50

        # 初始化距离矩阵
        if distance_matrix is not None:
            self._validate_matrix(distance_matrix)
            self.distance_matrix = distance_matrix
            self.filename = "Custom Matrix"
        else:
            data = read_tsp_file(filename)
            self.distance_matrix = np.array(data['distance_matrix'])
            self.filename = filename
            
        self.dimension = self.distance_matrix.shape[0]

        # 初始化操作符和种群
        self.operator = [
            naive_swap, chunk_swap, chunk_swap,
            naive_insert, greedy_insert, greedy_insert,
            greedy_insert, chunk_insert, chunk_insert,
            naive_reverse, naive_reverse,
            opt_swap_2, opt_swap_2,
            opt_swap_3, opt_swap_3
        ]
        self.init_fireworks = init_fireworks or []
        self.fireworks = []
        self.sparks = []
        self.best_seen = {'length': math.inf, 'tour': []}
        self.convergence = []

        # 打印参数
        self.print_parameters()

    def _validate_matrix(self, matrix: np.ndarray):
        """校验距离矩阵有效性"""
        if matrix.ndim != 2:
            raise ValueError("距离矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("距离矩阵必须是对称方阵")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("距离矩阵必须包含数值型数据")
        if np.any(matrix < 0):
            raise ValueError("距离矩阵不能包含负值")
            
    def print_parameters(self):
        """打印参数信息"""
        print("=" * 50)
        print(f"Algorithm: {self.tag}")
        print(f"Dataset: {self.filename}")
        print(f"Dimension: {self.dimension}")
        print(f"Verbose Mode: {self.verbose}")
        print(f"Boost Mode: {self.boost}")
        print("-" * 50)
        print(f"Epochs: {self.epoch}")
        print(f"Early Stop: {self.early_stop} iterations")
        print(f"Population Size: {self.size}")
        print(f"Alpha: {self.alpha}")
        print(f"Alpha_: {self.alpha_}")
        print(f"Firework Sparks Factor: {self.nspk_factor}")
        print(f"Amplitude: {self.amp}")
        print(f"Jump-out Threshold: {self.jumpout}")
        print("=" * 50)

    @timeit
    def solve(self):
        self.initialize_fireworks()
        self.convergence = []  # 重置记录器

        for epoch in range(self.epoch):
            boss_length, step = math.inf, 0
            while step <= self.early_stop:
                self.explode()
                self.mutate()
                self.select()

                self.convergence.append(self.best_seen["length"])

                if self.verbose:
                    print(f'{epoch+1} - selected with best seen {self.best_seen["length"]}')
                step += 1

                current_best_length = calculate_distance(self.best_seen['tour'], self.distance_matrix)
                if current_best_length < boss_length:
                    boss_length = current_best_length
                    step = 0
                    self.amp = max(self.amp / 1.1, 1)
                else:
                    self.amp = min(self.amp * 1.1, 100)
                # print(self.amp)

                if step == self.jumpout: 
                    # TODO: 重新初始化一半的烟花（精英仍保留）
                    pass
                    # self.fireworks = self.fireworks[self.size // 2:]
                    

                
            # Basin-hopping: 重新初始化种群
            self.initialize_fireworks()

        # 返回结果
        return self.tag, self.filename, self.best_seen['length'], self.best_seen['tour']

    def initialize_fireworks(self):
        self.fireworks.extend(self.init_fireworks)
        for _ in range(self.size - len(self.fireworks)):
            self.fireworks.append(
                list(np.random.permutation(list(range(1, self.dimension + 1)))))

    def calculate_firework_number(self):
        # 计算每个烟花的适应度排名
        distances = [calculate_distance(tour, self.distance_matrix) for tour in self.fireworks]
        sorted_indices = np.argsort(distances)  # 按距离从小到大排序
        ranks = np.zeros(self.size, dtype=int)
        for idx, rank in enumerate(sorted_indices):
            ranks[rank] = idx + 1  # 排名从1开始

        # 计算幂律分布
        total = sum(r ** (-self.alpha) for r in range(1, self.size + 1))
        num_spks = []
        for rank in ranks:
            lambda_r = self.nspk_factor * (float(rank) ** (-self.alpha)) / total
            num_spks.append(max(1, int(round(lambda_r))))
        # print(num_spks)
        return num_spks

    def explode(self):
        self.sparks = [[] for _ in range(len(self.fireworks))]
        num_spks = self.calculate_firework_number()

        # operator_big = [opt_swap_2, opt_swap_3, chunk_swap]
        # operator_mid = [chunk_insert, naive_reverse]
        # operator_small = [naive_swap, naive_insert]

        # amp_threshold_mid = (self.min_amp + self.max_amp) / 3     
        # amp_threshold_big = (self.min_amp + self.max_amp) * 2 / 3
        
        for i, (firework, spk_count) in enumerate(zip(self.fireworks, num_spks)):
            fw_dist = calculate_distance(firework, self.distance_matrix)
            for _ in range(spk_count):
                spark = random.choice(self.operator)(firework, self.distance_matrix)
                spark_dist = calculate_distance(spark, self.distance_matrix)

                if spark_dist < fw_dist:
                    self.sparks[i].append(spark)
                else:
                    # TODO: 能不能用距离，改用排名？
                    acceptance = math.exp(-(spark_dist - fw_dist) / self.amp) # 与amp成正比
                    if self.amp == 1:
                        print(acceptance)
                    if random.random() < acceptance:
                        self.sparks[i].append(spark)
            
    def mutate(self):
        pass

    def select(self):
        all_sparks = []
        for spk_list in self.sparks:
            all_sparks.extend(spk_list)
        candidates = self.fireworks + all_sparks
        distances = [calculate_distance(t, self.distance_matrix) for t in candidates]
        order = np.argsort(distances)
        self.best_seen['length'] = distances[order[0]]
        self.best_seen['tour'] = candidates[order[0]].copy()

        selected = [self.best_seen['tour']]

        sorted_candidates = [candidates[i] for i in order]

        remaining_candidates = sorted_candidates[1:]

        # 平均概率选择剩余的烟花
        k = self.size - len(selected)  # 还需要的个数
        if k > 0:
            chosen_indices = np.random.choice(
                len(remaining_candidates),
                size=k,
                replace=False
            )
            for idx in chosen_indices:
                selected.append(remaining_candidates[idx])

        # 排名概率选择剩余的烟花
        # ranks = np.arange(len(remaining_candidates))  # [0, 1, 2, ...]
        # weights = self.alpha_ ** ranks  # p(i) = alpha^i
        # weights_sum = np.sum(weights)
        # probs = weights / weights_sum

        # k = self.size - len(selected)  # 还需要的个数
        # if k > 0:
        #     chosen_indices = np.random.choice(
        #         len(remaining_candidates),
        #         size=k,
        #         replace=False,
        #         p=probs
        #     )
        #     for idx in chosen_indices:
        #         selected.append(remaining_candidates[idx])

        self.fireworks = selected

    def plot_convergence(self, window_size=100, max_iter=500, save_path=None):
        """
        绘制收敛曲线
        :param window_size: 移动平均窗口大小
        :param save_path: 图像保存路径
        """
        if not self.convergence:
            raise ValueError("No convergence data available")

        self.convergence = self.convergence[:max_iter]

        # 计算移动平均
        moving_avg = [
            np.mean(self.convergence[max(0, i-window_size):i+1]) 
            for i in range(len(self.convergence))
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence, 'b-', alpha=0.3, label='Instant Value')
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} steps)')
        plt.title(f'Convergence Curve\n{self.filename}')
        plt.xlabel('Iteration')
        plt.ylabel('Best Tour Length')
        
        # 添加关键点标注
        min_idx = np.argmin(self.convergence)
        plt.scatter(min_idx, self.convergence[min_idx], 
                   c='green', s=100, edgecolors='black', 
                   label=f'Global Minimum in First 500 Iterations({self.convergence[min_idx]:.2f})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
if __name__ == "__main__":
    filename = './data/eil76.tsp'
    
    dfwa = DFWAlgorithm(filename, epoch=3, early_stop=1000, verbose=True, size=15)
    tag, filename, best_length, best_tour = dfwa.solve()
    # 绘制结果
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    # dfwa.plot_convergence(save_path=f'{base_filename}.png')

    print(f"{tag} - Best Length: {best_length}")
    print(f"Best Tour: {best_tour}")