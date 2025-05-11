"""
FID评估脚本 - 计算生成图像与真实图像之间的Fréchet Inception Distance
"""

import os
import argparse
import torch
from pytorch_fid import fid_score
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='计算两个图像集合之间的FID分数')
    parser.add_argument('--real_path', type=str, default="/home/czar/ML/GAN/HGGAN/results/HGGAN/test_latest/images/real_image", 
                        help='真实图像的目录路径')
    parser.add_argument('--gen_path', type=str, default="/home/czar/ML/GAN/HGGAN/results/HGGAN/test_latest/images/synthesized_image", 
                        help='生成图像的目录路径')
    parser.add_argument('--batch_size', type=int, default=50, 
                        help='InceptionV3网络的批处理大小')
    parser.add_argument('--device', type=str, default=None, 
                        help='计算设备 (例如 "cuda:0" 或 "cpu")')
    parser.add_argument('--dims', type=int, default=2048, 
                        help='特征维度，InceptionV3的默认值为2048')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='数据加载的工作线程数')
    return parser.parse_args()

def check_paths(real_path, gen_path):
    """检查路径是否存在并包含图像"""
    real_dir = Path(real_path)
    gen_dir = Path(gen_path)
    
    if not real_dir.exists():
        raise ValueError(f"真实图像路径不存在: {real_path}")
    if not gen_dir.exists():
        raise ValueError(f"生成图像路径不存在: {gen_path}")
    
    # 检查每个目录中的图像数量
    real_imgs = [f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gen_imgs = [f for f in os.listdir(gen_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(real_imgs)} 张真实图像和 {len(gen_imgs)} 张生成图像")
    
    if len(real_imgs) < 100 or len(gen_imgs) < 100:
        print("警告: 每个目录建议至少有1000张图像以获得可靠的FID分数。当前图像数量可能导致不准确的结果。")
    
    return True

def calculate_fid(args):
    """计算FID分数"""
    print("开始计算FID分数...")
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    try:
        # 使用pytorch-fid计算FID
        fid_value = fid_score.calculate_fid_given_paths(
            [args.real_path, args.gen_path],
            batch_size=args.batch_size,
            device=device,
            dims=args.dims,
            num_workers=args.num_workers
        )
        
        print(f"\n======================================================")
        print(f"FID分数: {fid_value:.4f}")
        print(f"======================================================\n")
        
        # 将结果保存到文件
        result_file = os.path.join(os.path.dirname(args.gen_path), 'fid_result.txt')
        with open(result_file, 'w') as f:
            f.write(f"真实图像路径: {args.real_path}\n")
            f.write(f"生成图像路径: {args.gen_path}\n")
            f.write(f"FID分数: {fid_value:.4f}\n")
            f.write(f"批大小: {args.batch_size}\n")
            f.write(f"特征维度: {args.dims}\n")
        
        print(f"结果已保存至: {result_file}")
        
        return fid_value
        
    except Exception as e:
        print(f"计算FID时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    # 检查路径
    try:
        check_paths(args.real_path, args.gen_path)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    # 计算FID
    calculate_fid(args)

if __name__ == "__main__":
    main()