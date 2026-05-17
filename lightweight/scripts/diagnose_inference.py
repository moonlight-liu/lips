"""
诊断推理问题:
1. 检查 ckpt 内容
2. 检查权重加载是否成功
3. 用真实样本和随机噪声对比模型输出
4. 检查模型梯度统计
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import torch
import numpy as np
from models import build_model
from data import AVLip
import argparse


def diagnose(ckpt_path, real_path, fake_path):
    print("=" * 70)
    print("LipFD 推理诊断脚本")
    print("=" * 70)

    # 1. 检查 ckpt 文件
    print("\n[1] 检查 checkpoint 文件...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"  checkpoint keys: {list(state_dict.keys())}")
    if "model" in state_dict:
        model_sd = state_dict["model"]
        print(f"  'model' 字典中有 {len(model_sd)} 个参数")
        if "epoch" in state_dict:
            print(f"  epoch: {state_dict['epoch']}")
        if "total_steps" in state_dict:
            print(f"  total_steps: {state_dict['total_steps']}")
        # 看看几个关键参数的范围，判断是否是初始化的还是训练过的
        sample_keys = ["backbone.fc.weight", "backbone.fc.bias",
                       "backbone.get_weight.0.weight", "conv1.weight"]
        for k in sample_keys:
            if k in model_sd:
                v = model_sd[k]
                print(f"  {k}: shape={tuple(v.shape)}, "
                      f"mean={v.float().mean().item():.4f}, "
                      f"std={v.float().std().item():.4f}, "
                      f"min={v.float().min().item():.4f}, "
                      f"max={v.float().max().item():.4f}")
    else:
        print(f"  ⚠️  没有 'model' key，直接是 state_dict")

    # 2. 加载模型
    print("\n[2] 加载模型...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model("CLIP:ViT-L/14", device=device)
    sd_to_load = state_dict["model"] if "model" in state_dict else state_dict
    incompatible = model.load_state_dict(sd_to_load, strict=False)
    print(f"  missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:5]}")
    print(f"  unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:5]}")
    model.eval()
    model.to(device)

    # 3. 用一些不同的真实输入测试模型
    print("\n[3] 用验证集真实数据测试...")

    class Opt:
        pass
    opt = Opt()
    opt.data_label = "val"
    opt.real_list_path = real_path
    opt.fake_list_path = fake_path
    dataset = AVLip(opt)
    print(f"  数据集大小: {len(dataset)}")

    print(f"\n  逐张推理（看 pre-sigmoid logits 和 sigmoid score）：")
    print(f"  {'idx':<5}{'label':<8}{'logit':<15}{'score':<15}{'path'}")
    for i in range(min(len(dataset), 10)):
        img, crops, label = dataset[i]
        img_b = img.unsqueeze(0).to(device)
        crops_b = [[c.unsqueeze(0).to(device) for c in sub] for sub in crops]
        with torch.no_grad():
            feat = model.get_features(img_b)
            logit = model(crops_b, feat)[0]
            score = logit.sigmoid()
        print(f"  {i:<5}{label:<8}{logit.item():<15.4f}{score.item():<15.4f}{dataset.total_list[i]}")

    # 4. 用纯随机噪声测试 - 如果模型也输出 0.96，说明模型已经塌缩
    print("\n[4] 用随机噪声测试（如果输出和真实数据相同，说明模型已塌缩）...")
    torch.manual_seed(42)
    for trial in range(3):
        rand_img = torch.randn(1, 3, 1120, 1120).to(device)
        rand_crops = [[torch.randn(1, 3, 224, 224).to(device) for _ in range(5)]
                      for _ in range(3)]
        with torch.no_grad():
            feat = model.get_features(rand_img)
            logit = model(rand_crops, feat)[0]
            score = logit.sigmoid()
        print(f"  随机输入 {trial}: logit={logit.item():.4f}, score={score.item():.4f}")

    # 5. 测试 fc 层输出对 feature 变化的敏感度
    print("\n[5] 测试模型对输入变化的敏感度...")
    rand_img = torch.randn(1, 3, 1120, 1120).to(device)
    rand_crops = [[torch.randn(1, 3, 224, 224).to(device) for _ in range(5)]
                  for _ in range(3)]
    scores = []
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        with torch.no_grad():
            scaled_crops = [[c * scale for c in sub] for sub in rand_crops]
            feat = model.get_features(rand_img * scale)
            logit = model(scaled_crops, feat)[0]
        scores.append(logit.sigmoid().item())
    print(f"  不同输入 scale 下的 score: {scores}")
    print(f"  score 范围: {max(scores) - min(scores):.4f}")
    if max(scores) - min(scores) < 0.05:
        print(f"  ⚠️  输出对输入几乎不敏感 -> 模型可能已塌缩，或 fc 层输出 saturated")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./checkpoints/ckpt.pth")
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    args = parser.parse_args()
    diagnose(args.ckpt, args.real_list_path, args.fake_list_path)
