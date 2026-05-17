"""
测量模型的参数量、FLOPs 和内存占用

使用方法:
    python scripts/measure_model.py --model original
    python scripts/measure_model.py --model lightweight

输出:
    - 总参数量
    - 各模块参数量分布
    - FLOPs (浮点运算次数)
    - 模型大小
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import argparse
from thop import profile, clever_format


def measure_original_model():
    """测量原始 LipFD 模型"""
    print("=" * 60)
    print("测量原始模型 (ViT-L/14 + ResNet50)")
    print("=" * 60)

    from models.LipFD import LipFD

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LipFD(name="ViT-L/14", device=device)
    model.to(device)
    model.eval()

    # 1. 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n【参数量统计】")
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 2. 分模块统计
    print(f"\n【模块参数量分布】")

    # CLIP Encoder
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"CLIP ViT-L/14: {encoder_params:,} ({encoder_params/1e6:.2f}M) - {encoder_params/total_params*100:.1f}%")

    # Region Awareness (Backbone)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"ResNet50 (Region Awareness): {backbone_params:,} ({backbone_params/1e6:.2f}M) - {backbone_params/total_params*100:.1f}%")

    # Conv1
    conv1_params = sum(p.numel() for p in model.conv1.parameters())
    print(f"Conv1 (下采样层): {conv1_params:,} ({conv1_params/1e3:.2f}K) - {conv1_params/total_params*100:.1f}%")

    # 3. 计算 FLOPs
    print(f"\n【计算复杂度】")
    print("正在计算 FLOPs (这可能需要一些时间)...")

    try:
        # 准备输入
        # 完整图像用于提取全局特征
        img = torch.randn(1, 3, 1120, 1120).to(device)

        # 区域图像用于 Region Awareness
        # 3个区域 × 5帧
        crops = [[torch.randn(1, 3, 224, 224).to(device) for _ in range(5)] for _ in range(3)]

        # 先计算全局特征的 FLOPs
        with torch.no_grad():
            global_feature = model.get_features(img)

        # 使用 thop 计算 backbone 的 FLOPs
        flops, params = profile(model.backbone, inputs=(crops, global_feature), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        print(f"FLOPs (Region Awareness): {flops_formatted}")
        print(f"注意: 这只是 Region Awareness 部分的 FLOPs")
        print(f"CLIP Encoder 的 FLOPs 较难精确测量，估计约为 30-50 GFLOPs")

    except Exception as e:
        print(f"FLOPs 计算失败: {e}")
        print("提示: 请确保 thop 已安装，并且模型与输入位于同一设备")

    # 4. 模型大小
    print(f"\n【模型大小】")
    try:
        # 保存模型到临时文件
        temp_path = "/tmp/lipfd_original.pth"
        torch.save(model.state_dict(), temp_path)
        model_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        print(f"模型文件大小: {model_size:.2f} MB")
        os.remove(temp_path)
    except Exception as e:
        print(f"无法计算模型大小: {e}")

    print("\n" + "=" * 60)

    return {
        'total_params': total_params,
        'encoder_params': encoder_params,
        'backbone_params': backbone_params,
        'model_size_mb': model_size if 'model_size' in locals() else None
    }


def measure_lightweight_model():
    """测量轻量化模型"""
    print("=" * 60)
    print("测量轻量化模型 (ViT-B/16 + ResNet34)")
    print("=" * 60)

    # 检查轻量化模型是否存在
    lightweight_model_path = os.path.join(os.path.dirname(__file__), '../models/LipFD_light.py')
    if not os.path.exists(lightweight_model_path):
        print("\n⚠️  轻量化模型文件不存在!")
        print(f"请先创建: {lightweight_model_path}")
        print("\n提示: 你可以先使用已有的 models/LipFD_fast.py")

        # 尝试使用 LipFD_fast
        try:
            from models.LipFD_fast import LipFD_Fast
            print("\n使用 models/LipFD_fast.py 进行测量...")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = LipFD_Fast(name="ViT-B/16", device=device)
            model.to(device)
            model.eval()

            # 参数量统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n【参数量统计】")
            print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

            # 分模块统计
            print(f"\n【模块参数量分布】")

            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            print(f"CLIP ViT-B/16: {encoder_params:,} ({encoder_params/1e6:.2f}M) - {encoder_params/total_params*100:.1f}%")

            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            print(f"ResNet34 (Region Awareness): {backbone_params:,} ({backbone_params/1e6:.2f}M) - {backbone_params/total_params*100:.1f}%")

            conv1_params = sum(p.numel() for p in model.conv1.parameters())
            print(f"Conv1 (下采样层): {conv1_params:,} ({conv1_params/1e3:.2f}K) - {conv1_params/total_params*100:.1f}%")

            print("\n" + "=" * 60)

            return {
                'total_params': total_params,
                'encoder_params': encoder_params,
                'backbone_params': backbone_params
            }

        except Exception as e:
            print(f"\n❌ 加载 LipFD_fast 失败: {e}")
            return None

    # TODO: 实现轻量化模型的测量
    print("轻量化模型测量功能待实现...")
    return None


def compare_models(original_stats, lightweight_stats):
    """对比两个模型"""
    if original_stats is None or lightweight_stats is None:
        print("\n⚠️  无法对比: 缺少模型统计数据")
        return

    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)

    print(f"\n{'指标':<30} {'原始模型':<20} {'轻量化模型':<20} {'减少比例':<15}")
    print("-" * 85)

    # 总参数量
    orig_total = original_stats['total_params']
    light_total = lightweight_stats['total_params']
    reduction = (1 - light_total / orig_total) * 100
    print(f"{'总参数量 (M)':<30} {orig_total/1e6:<20.2f} {light_total/1e6:<20.2f} {reduction:<15.1f}%")

    # Encoder
    orig_enc = original_stats['encoder_params']
    light_enc = lightweight_stats['encoder_params']
    enc_reduction = (1 - light_enc / orig_enc) * 100
    print(f"{'CLIP Encoder (M)':<30} {orig_enc/1e6:<20.2f} {light_enc/1e6:<20.2f} {enc_reduction:<15.1f}%")

    # Backbone
    orig_back = original_stats['backbone_params']
    light_back = lightweight_stats['backbone_params']
    back_reduction = (1 - light_back / orig_back) * 100
    print(f"{'Region Awareness (M)':<30} {orig_back/1e6:<20.2f} {light_back/1e6:<20.2f} {back_reduction:<15.1f}%")

    print("\n" + "=" * 60)

    # 总结
    print("\n【总结】")
    print(f"✓ 总参数量减少: {reduction:.1f}%")
    print(f"✓ 从 {orig_total/1e6:.1f}M 减少到 {light_total/1e6:.1f}M")
    print(f"✓ 节省了 {(orig_total - light_total)/1e6:.1f}M 参数")


def main():
    parser = argparse.ArgumentParser(description='测量 LipFD 模型的参数量和计算复杂度')
    parser.add_argument('--model', type=str, default='original',
                        choices=['original', 'lightweight', 'both'],
                        help='要测量的模型: original, lightweight, 或 both')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LipFD 模型性能测量工具")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")

    original_stats = None
    lightweight_stats = None

    if args.model in ['original', 'both']:
        try:
            original_stats = measure_original_model()
        except Exception as e:
            print(f"\n❌ 测量原始模型失败: {e}")
            import traceback
            traceback.print_exc()

    if args.model in ['lightweight', 'both']:
        try:
            lightweight_stats = measure_lightweight_model()
        except Exception as e:
            print(f"\n❌ 测量轻量化模型失败: {e}")
            import traceback
            traceback.print_exc()

    # 如果两个模型都测量了，进行对比
    if args.model == 'both' and original_stats and lightweight_stats:
        compare_models(original_stats, lightweight_stats)

    print("\n✅ 测量完成!")
    print("\n💡 提示:")
    print("  - 将这些数据记录到 EXPERIMENT_RESULTS.md 中")
    print("  - 这些数据将用于答辩时的对比展示")


if __name__ == '__main__':
    main()
