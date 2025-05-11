"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import random
import torch
import numpy as np
from collections import OrderedDict, deque
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.replay_buffer import ReplayBuffer
from trainers.pix2pix_trainer import Pix2PixTrainer

# 修改主训练循环以使用 Replay Buffer
def main():
    # 解析选项
    opt = TrainOptions().parse()
    
    # 打印选项帮助调试
    print(' '.join(sys.argv))
    
    # 加载数据集
    dataloader = data.create_dataloader(opt)
    
    # 创建 Replay Buffer
    if opt.use_replay_buffer:
        replay_buffer = ReplayBuffer(max_size=opt.replay_buffer_size, 
                                    replace_prob=opt.replay_replace_prob)
    
    # 创建模型训练器
    trainer = Pix2PixTrainer(opt)
    
    # 创建迭代计数工具
    iter_counter = IterationCounter(opt, len(dataloader))
    
    # 创建可视化工具
    visualizer = Visualizer(opt)
    
    
    # 预填充缓冲区
    if opt.use_replay_buffer:
        print("预填充 Replay Buffer...")
        prefill_count = min(1000, opt.replay_buffer_size)
        prefill_iter = iter(dataloader)
        for _ in range(prefill_count):
            try:
                data_i = next(prefill_iter)
                replay_buffer.add(data_i)
            except StopIteration:
                break
        print(f"Replay Buffer 已预填充 {replay_buffer.size()} 批次数据")
    
    # 主训练循环
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            
            # 添加新数据到缓冲区
            if opt.use_replay_buffer:
                replay_buffer.add(data_i)
            
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            
            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['label']),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            
            if opt.use_replay_buffer:
                for j in range(opt.sample_iter):
                    # 从缓冲区采样数据
                    sampled_data = replay_buffer.sample()
                    if sampled_data is not None:
                        # 训练生成器和判别器
                        if j % opt.D_steps_per_G == 0:
                            trainer.run_generator_one_step(sampled_data)
                        trainer.run_discriminator_one_step(sampled_data)
            
            # 可视化和打印
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                              losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
                
                # 打印 Replay Buffer 统计信息
                if opt.use_replay_buffer:
                    buffer_stats = replay_buffer.get_stats()
                    print(f"Replay Buffer: 大小={buffer_stats['current_size']}/{opt.replay_buffer_size}, "
                          f"添加={buffer_stats['additions']}, 采样={buffer_stats['samples']}, "
                          f"替换={buffer_stats['replacements']}")
            
            # 保存模型
            if iter_counter.needs_saving():
                print('保存最新模型 (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
        
        # 更新学习率
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()
        
        # 定期保存模型
        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('在 epoch %d, iter %d 保存模型' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
    
    print('训练成功完成')
    
if __name__ == "__main__":
    main()