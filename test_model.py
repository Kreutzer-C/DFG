import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
from torch.utils.data import DataLoader
from dataloaders import MyDataset
from models import get_model
from utils import mean_dice_new, mean_asd
import numpy as np
from tqdm import tqdm
import json
import cv2

# ── Visualisation colour palette ─────────────────────────────────────────────
LABEL_COLORS_HEX = [
    None,        # 0: background
    "#80AE80",   # 1: Spleen       — green
    "#F1D691",   # 2: Right Kidney — yellow
    "#B17A65",   # 3: Left Kidney  — brown-red
    "#6FB8D2",   # 4: Liver        — blue
]
LABEL_ALPHA       = 0.35
CONTOUR_THICKNESS = 1

def _hex_to_bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def overlay_labels(bgr, label):
    result = bgr.copy().astype(np.float32)
    for cls_idx, hex_color in enumerate(LABEL_COLORS_HEX):
        if hex_color is None:
            continue
        mask = (label == cls_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue
        bgr_color = _hex_to_bgr(hex_color)
        color_layer = np.zeros_like(result)
        color_layer[mask == 1] = bgr_color
        result = np.where(
            mask[:, :, None] == 1,
            result * (1 - LABEL_ALPHA) + color_layer * LABEL_ALPHA,
            result,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, bgr_color, CONTOUR_THICKNESS, cv2.LINE_AA)
    return result.clip(0, 255).astype(np.uint8)

def save_vis_slices(sample_dict, domain_tag, vis_dir):
    for patient_id in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[patient_id], key=lambda x: x[2])
        case_dir = os.path.join(vis_dir, f"{domain_tag}_{patient_id}")
        os.makedirs(case_dir, exist_ok=True)
        for pred_label, _gt, slice_idx, img_cpu in slices:
            img_np = img_cpu.numpy()
            mid = img_np.shape[0] // 2
            gray = img_np[mid]
            lo, hi = gray.min(), gray.max()
            gray = (gray - lo) / (hi - lo + 1e-8)
            img_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            bgr = overlay_labels(bgr, pred_label.numpy().astype(np.int32))
            bgr = cv2.flip(bgr, 1)
            fname = f"slice_{slice_idx:04d}_pred.png"
            cv2.imwrite(os.path.join(case_dir, fname), bgr)
        print(f"  Vis saved -> {case_dir}  ({len(slices)} slices)")
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, opt):
    """加载训练好的模型"""
    model = get_model(opt)
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    model = model.to(opt['gpu_id'])
    model.eval()
    return model

@torch.no_grad()
def test_model(model, test_dataloader, opt, save_vis=False):
    """测试模型性能"""
    sample_dict = {}
    
    test_iterator = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")
    for images, segs, names in test_iterator:
        images = images.to(opt['gpu_id'])
        segs = segs.to(opt['gpu_id'])
        
        _, predict = model(images)
        
        for i, name in enumerate(names):
            sample_name, index = name.split('_')[0], int(name.split('_')[1])
            img_cpu = images[i].cpu() if save_vis else None
            sample_dict[sample_name] = sample_dict.get(sample_name, []) + [
                (predict[i].detach().cpu().argmax(dim=0), segs[i].detach().cpu(), index, img_cpu)
            ]
    
    # 整理预测结果
    pred_results_list = []
    gt_segs_list = []
    for k in sample_dict.keys():
        sample_dict[k].sort(key=lambda ele: ele[2])
        preds = []
        targets = []
        for pred, target, _, _img in sample_dict[k]:
            if target.sum() == 0:
                continue
            preds.append(pred)
            targets.append(target)
        pred_results_list.append(torch.stack(preds, dim=-1))
        gt_segs_list.append(torch.stack(targets, dim=-1))
    
    # 计算指标
    dice_metrics = mean_dice_new(pred_results_list, gt_segs_list, opt['num_classes'], opt['organ_list'])
    asd_metrics = mean_asd(pred_results_list, gt_segs_list, opt['num_classes'], opt['organ_list'])
    
    return dice_metrics, asd_metrics

def main():
    # 基础配置
    opt = {
        'data_root': '/opt/data/private/DFG/datasets/Abdomen_Data_new',
        'gpu_id': 0,
        'batch_size': 16,
        'num_workers': 8,
        'arch': 'UNet',
        'use_prototype': False,
        'update_prototype': False,
        'input_dim': 3,
        'feat_dim': 32,
        'output_dim': 64,
        'num_classes': 5,
        'organ_list': ['Spleen', 'R.Kidney', 'L.Kidney', 'Liver'],
        'save_vis': True,
        'vis_dir': '/opt/data/private/DFG/results/vis_results'
    }
    
    # 模型路径
    models = {
        'CT': '/opt/data/private/DFG/results/Target_Adapt/SAM_UNet_Abdomen_CT2MR/exp_0_time_2026-03-06 15:16:11/saved_models/best_model_step_5_dice_0.8604.pth',
        'MRI': '/opt/data/private/DFG/results/Target_Adapt/SAM_UNet_Abdomen_MR2CT/exp_2_time_2026-03-05 20:19:19/saved_models/model_step_100_dice_0.5724.pth'
    }
    
    # 测试域
    domains = ['CT', 'MRI']
    
    results = {}
    
    # 测试所有模型和域的组合
    for model_name, model_path in models.items():
        print(f"\n{'='*80}")
        print(f"测试模型: {model_name} (训练域)")
        print(f"模型路径: {model_path}")
        print(f"{'='*80}\n")
        
        # 加载模型
        model = load_model(model_path, opt)
        
        results[model_name] = {}
        
        for test_domain in domains:
            print(f"\n{'-'*80}")
            if test_domain == model_name:
                print(f"测试: In-Domain ({test_domain})")
                domain_type = 'in-domain'
            else:
                print(f"测试: Cross-Domain ({model_name} -> {test_domain})")
                domain_type = 'cross-domain'
            print(f"{'-'*80}\n")
            
            # 创建测试数据加载器
            test_dataloader = DataLoader(
                MyDataset(opt['data_root'], [test_domain], phase='test', split_train=False),
                batch_size=opt['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=opt['num_workers']
            )
            
            print(f"测试数据集大小: {len(test_dataloader)} batches")
            
            # 测试模型
            dice_metrics, asd_metrics = test_model(model, test_dataloader, opt, save_vis=opt.get("save_vis", False))
            
            # 保存结果
            results[model_name][test_domain] = {
                'domain_type': domain_type,
                'dice': dice_metrics,
                'asd': asd_metrics
            }
            
            # 打印结果
            print(f"\n{'*'*60}")
            print(f"结果 - {model_name} 模型 on {test_domain} 数据 ({domain_type})")
            print(f"{'*'*60}")
            print("\nDice 分数:")
            for key, value in dice_metrics.items():
                print(f"  {key}: {value:.4f}")
            print("\nASD (平均表面距离):")
            for key, value in asd_metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"{'*'*60}\n")
    
    # 保存完整结果到文件
    output_dir = '/opt/data/private/DFG/results/test_results-SAM'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式结果
    json_results = {}
    for model_name in results:
        json_results[model_name] = {}
        for test_domain in results[model_name]:
            json_results[model_name][test_domain] = {
                'domain_type': results[model_name][test_domain]['domain_type'],
                'dice': {k: float(v) for k, v in results[model_name][test_domain]['dice'].items()},
                'asd': {k: float(v) for k, v in results[model_name][test_domain]['asd'].items()}
            }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # 生成汇总报告
    print("\n" + "="*100)
    print("测试结果汇总报告")
    print("="*100 + "\n")
    
    for model_name in ['CT', 'MRI']:
        print(f"\n{'#'*100}")
        print(f"# 模型: {model_name}")
        print(f"{'#'*100}\n")
        
        # In-domain 结果
        print(f"\n【In-Domain 性能】 - {model_name} 模型在 {model_name} 数据上")
        print("-" * 80)
        in_domain_result = results[model_name][model_name]
        print("Dice 分数:")
        for key, value in in_domain_result['dice'].items():
            print(f"  {key:20s}: {value:.4f}")
        print("\nASD (平均表面距离):")
        for key, value in in_domain_result['asd'].items():
            print(f"  {key:20s}: {value:.4f}")
        
        # Cross-domain 结果
        cross_domain = 'MRI' if model_name == 'CT' else 'CT'
        print(f"\n【Cross-Domain 性能】 - {model_name} 模型在 {cross_domain} 数据上")
        print("-" * 80)
        cross_domain_result = results[model_name][cross_domain]
        print("Dice 分数:")
        for key, value in cross_domain_result['dice'].items():
            print(f"  {key:20s}: {value:.4f}")
        print("\nASD (平均表面距离):")
        for key, value in cross_domain_result['asd'].items():
            print(f"  {key:20s}: {value:.4f}")
        
        # 性能下降分析
        print(f"\n【性能下降分析】")
        print("-" * 80)
        print("Dice 分数下降:")
        for key in in_domain_result['dice'].keys():
            in_val = in_domain_result['dice'][key]
            cross_val = cross_domain_result['dice'][key]
            diff = in_val - cross_val
            percent = (diff / in_val * 100) if in_val > 0 else 0
            print(f"  {key:20s}: {diff:+.4f} ({percent:+.2f}%)")
        print("\nASD 变化:")
        for key in in_domain_result['asd'].keys():
            in_val = in_domain_result['asd'][key]
            cross_val = cross_domain_result['asd'][key]
            diff = cross_val - in_val
            percent = (diff / in_val * 100) if in_val > 0 else 0
            print(f"  {key:20s}: {diff:+.4f} ({percent:+.2f}%)")
        print()
    
    print("\n" + "="*100)
    print(f"测试完成! 结果已保存到: {output_dir}/test_results.json")
    print("="*100 + "\n")
    
    # 保存汇总报告
    with open(os.path.join(output_dir, 'test_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("测试结果汇总报告\n")
        f.write("="*100 + "\n\n")
        
        for model_name in ['CT', 'MRI']:
            f.write(f"\n{'#'*100}\n")
            f.write(f"# 模型: {model_name}\n")
            f.write(f"{'#'*100}\n\n")
            
            # In-domain 结果
            f.write(f"\n【In-Domain 性能】 - {model_name} 模型在 {model_name} 数据上\n")
            f.write("-" * 80 + "\n")
            in_domain_result = results[model_name][model_name]
            f.write("Dice 分数:\n")
            for key, value in in_domain_result['dice'].items():
                f.write(f"  {key:20s}: {value:.4f}\n")
            f.write("\nASD (平均表面距离):\n")
            for key, value in in_domain_result['asd'].items():
                f.write(f"  {key:20s}: {value:.4f}\n")
            
            # Cross-domain 结果
            cross_domain = 'MRI' if model_name == 'CT' else 'CT'
            f.write(f"\n【Cross-Domain 性能】 - {model_name} 模型在 {cross_domain} 数据上\n")
            f.write("-" * 80 + "\n")
            cross_domain_result = results[model_name][cross_domain]
            f.write("Dice 分数:\n")
            for key, value in cross_domain_result['dice'].items():
                f.write(f"  {key:20s}: {value:.4f}\n")
            f.write("\nASD (平均表面距离):\n")
            for key, value in cross_domain_result['asd'].items():
                f.write(f"  {key:20s}: {value:.4f}\n")
            
            # 性能下降分析
            f.write(f"\n【性能下降分析】\n")
            f.write("-" * 80 + "\n")
            f.write("Dice 分数下降:\n")
            for key in in_domain_result['dice'].keys():
                in_val = in_domain_result['dice'][key]
                cross_val = cross_domain_result['dice'][key]
                diff = in_val - cross_val
                percent = (diff / in_val * 100) if in_val > 0 else 0
                f.write(f"  {key:20s}: {diff:+.4f} ({percent:+.2f}%)\n")
            f.write("\nASD 变化:\n")
            for key in in_domain_result['asd'].keys():
                in_val = in_domain_result['asd'][key]
                cross_val = cross_domain_result['asd'][key]
                diff = cross_val - in_val
                percent = (diff / in_val * 100) if in_val > 0 else 0
                f.write(f"  {key:20s}: {diff:+.4f} ({percent:+.2f}%)\n")
            f.write("\n")

if __name__ == '__main__':
    main()
