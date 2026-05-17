"""
测试模型的推理速度

使用方法:
    python scripts/benchmark_inference.py --model original
    python scripts/benchmark_inference.py --model lightweight --batch_size 16

输出:
    - 不同 batch size 下的推理时间
    - FPS (每秒处理帧数)
    - GPU/CPU 推理对比
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import argparse
import time
import numpy as np
from tqdm import tqdm


def benchmark_model(model, device, batch_sizes=[1, 4, 8, 16], num_warmup=10, num_iterations=100):
    """
    测试模型的推理速度

    Args:
        model: 要测试的模型
        device: 运行设备
        batch_sizes: 要测试的 batch size 列表
        num_warmup: 预热次数
        num_iterations: 测试迭代次数
    """
    model.eval()
    results = []

    print(f"\n{'Batch Size':<12} {'平均时间(ms)':<15} {'标准差(ms)':<15} {'FPS':<10}")
    print("-" * 60)

    for batch_size in batch_sizes:
        # 准备输入数据
        img = torch.randn(batch_size, 3, 1120, 1120).to(device)
        crops = [[torch.randn(batch_size, 3, 224, 224).to(device) for _ in range(5)] for _ in range(3)]

        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                global_feature = model.get_features(img)
                _ = model.forward(crops, global_feature)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

        # 正式测试
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                global_feature = model.get_features(img)
                _ = model.forward(crops, global_feature)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()

                times.append((end_time - start_time) * 1000)  # 转换为毫秒

        # 计算统计数据
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / mean_time * batch_size  # 每秒处理的样本数

        print(f"{batch_size:<12} {mean_time:<15.2f} {std_time:<15.2f} {fps:<10.1f}")

        results.append({
            'batch_size': batch_size,
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'fps': fps
        })

    return results


def benchmark_original_model(device, batch_sizes):
    """测试原始模型"""
    print("=" * 60)
    print("测试原始模型推理速度 (ViT-L/14 + ResNet50)")
    print("=" * 60)
    print(f"设备: {device}")

    from models.LipFD import LipFD

    model = LipFD(name="ViT-L/14", device=device)
    model.eval()

    results = benchmark_model(model, device, batch_sizes)

    return results


def benchmark_lightweight_model(device, batch_sizes):
    """测试轻量化模型"""
    print("=" * 60)
    print("测试轻量化模型推理速度 (ViT-B/16 + ResNet34)")
    print("=" * 60)
    print(f"设备: {device}")

    try:
        from models.LipFD_fast import LipFD_Fast
        model = LipFD_Fast(name="ViT-B/16", device=device)
        model.eval()

        results = benchmark_model(model, device, batch_sizes)
        return results

    except Exception as e:
        print(f"\n❌ 加载轻量化模型失败: {e}")
        return None


def compare_results(original_results, lightweight_results):
    """对比两个模型的推理速度"""
    if original_results is None or lightweight_results is None:
        print("\n⚠️  无法对比: 缺少测试结果")
        return

    print("\n" + "=" * 60)
    print("推理速度对比")
    print("=" * 60)

    print(f"\n{'Batch Size':<12} {'原始(ms)':<15} {'轻量化(ms)':<15} {'加速比':<10}")
    print("-" * 60)

    for orig, light in zip(original_results, lightweight_results):
        if orig['batch_size'] == light['batch_size']:
            speedup = orig['mean_time_ms'] / light['mean_time_ms']
            print(f"{orig['batch_size']:<12} {orig['mean_time_ms']:<15.2f} "
                  f"{light['mean_time_ms']:<15.2f} {speedup:<10.2f}x")

    print("\n" + "=" * 60)

    # 计算平均加速比
    speedups = [orig['mean_time_ms'] / light['mean_time_ms']
                for orig, light in zip(original_results, lightweight_results)
                if orig['batch_size'] == light['batch_size']]
    avg_speedup = np.mean(speedups)

    print(f"\n【总结】")
    print(f"✓ 平均加速比: {avg_speedup:.2f}x")
    print(f"✓ 轻量化模型比原始模型快 {(avg_speedup - 1) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='测试 LipFD 模型的推理速度')
    parser.add_argument('--model', type=str, default='original',
                        choices=['original', 'lightweight', 'both'],
                        help='要测试的模型')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='运行设备')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16],
                        help='要测试的 batch size')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='测试迭代次数')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='预热次数')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，切换到 CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("LipFD 模型推理速度测试工具")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"测试配置:")
    print(f"  - Batch sizes: {args.batch_sizes}")
    print(f"  - 预热次数: {args.num_warmup}")
    print(f"  - 测试迭代: {args.num_iterations}")
    print("=" * 60 + "\n")

    original_results = None
    lightweight_results = None

    if args.model in ['original', 'both']:
        try:
            original_results = benchmark_original_model(device, args.batch_sizes)
        except Exception as e:
            print(f"\n❌ 测试原始模型失败: {e}")
            import traceback
            traceback.print_exc()

    if args.model in ['lightweight', 'both']:
        try:
            lightweight_results = benchmark_lightweight_model(device, args.batch_sizes)
        except Exception as e:
            print(f"\n❌ 测试轻量化模型失败: {e}")
            import traceback
            traceback.print_exc()

    # 对比结果
    if args.model == 'both' and original_results and lightweight_results:
        compare_results(original_results, lightweight_results)

    print("\n✅ 测试完成!")
    print("\n💡 提示:")
    print("  - 将这些数据记录到 EXPERIMENT_RESULTS.md 中")
    print("  - GPU 推理时间更准确，建议使用 --device cuda")
    print("  - 可以尝试不同的 batch size 找到最优配置")


if __name__ == '__main__':
    main()
