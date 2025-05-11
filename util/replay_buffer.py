import sys
import time
import random
import torch
import numpy as np
from collections import OrderedDict, deque
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer


# Replay Buffer 实现
class ReplayBuffer:
    def __init__(self, max_size=100, replace_prob=0.1):
        """
        初始化 Replay Buffer
        
        参数:
            max_size: 缓冲区最大容量
            replace_prob: 替换已有数据的概率
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.replace_prob = replace_prob
        self.stats = {
            "additions": 0,
            "samples": 0,
            "replacements": 0,
        }
        print(f"初始化 Replay Buffer，容量: {max_size}，替换概率: {replace_prob}")
    
    def add(self, data):
        """将数据添加到缓冲区"""
        # 深复制张量以避免引用问题
        copied_data = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                copied_data[k] = v.clone().detach()
            else:
                copied_data[k] = v
                
        # 如果缓冲区已满，根据替换概率决定是否替换
        if len(self.buffer) == self.max_size:
            if random.random() < self.replace_prob:
                # 随机替换一个数据点
                idx = random.randint(0, self.max_size - 1)
                self.buffer[idx] = copied_data
                self.stats["replacements"] += 1
        else:
            self.buffer.append(copied_data)
            
        self.stats["additions"] += 1
    
    def sample(self, batch_size=None):
        """从缓冲区采样数据"""
        if not self.buffer:
            return None
            
        self.stats["samples"] += 1
        
        if batch_size is None or batch_size >= len(self.buffer):
            return random.choice(self.buffer)
        else:
            return random.choice(self.buffer)
    
    def size(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def get_stats(self):
        """获取缓冲区统计信息"""
        self.stats["current_size"] = len(self.buffer)
        return self.stats
