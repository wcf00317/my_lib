import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import argparse

# =========================================================================
# 1. 环境与路径设置
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
nyu_path = os.path.join(current_dir, '../nyu')
if nyu_path not in sys.path: sys.path.append(nyu_path)

from aspp import DeepLabHead
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_device

# =========================================================================
# 2. 严格复刻 Trainer
# =========================================================================
class Citytrainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, 
                 decoders, rep_grad, multi_input, optim_param, 
                 scheduler_param, **kwargs):
        super(Citytrainer, self).__init__(task_dict=task_dict, 
                                          weighting=weighting, 
                                          architecture=architecture, 
                                          encoder_class=encoder_class, 
                                          decoders=decoders,
                                          rep_grad=rep_grad,
                                          multi_input=multi_input,
                                          optim_param=optim_param,
                                          scheduler_param=scheduler_param,
                                          **kwargs)

    def process_preds(self, preds):
        # 官方逻辑: 强制对齐到 128x256
        img_size = (128, 256)
        for task in self.task_name:
            preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
        return preds

# =========================================================================
# 3. 数据集定义
# =========================================================================
class CityscapesC_Dataset(Dataset):
    def __init__(self, images_dir, gt_root):
        self.gt_root = gt_root
        self.img_paths = []
        
        if os.path.exists(images_dir):
            subfolders = sorted([d for d in os.listdir(images_dir) 
                               if os.path.isdir(os.path.join(images_dir, d))])
            if len(subfolders) > 0:
                for city in subfolders:
                    city_path = os.path.join(images_dir, city)
                    files = sorted([f for f in os.listdir(city_path) if f.endswith('.png')])
                    for f in files:
                        self.img_paths.append(os.path.join(city_path, f))
            else:
                files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
                for f in files:
                    self.img_paths.append(os.path.join(images_dir, f))
        
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Resize (256, 128) -> Tensor
        img_pil = Image.open(img_path).convert('RGB')
        img_resized = img_pil.resize((256, 128), resample=Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        # GT Reading
        label_path = os.path.join(self.gt_root, 'val', 'label', f'{idx}.npy')
        depth_path = os.path.join(self.gt_root, 'val', 'depth', f'{idx}.npy')

        try:
            label = torch.from_numpy(np.load(label_path)).long()
            if os.path.exists(depth_path):
                depth = torch.from_numpy(np.load(depth_path)).float()
            else:
                depth = torch.zeros_like(label).float()
        except Exception:
            label = torch.zeros((128, 256)).long()
            depth = torch.zeros((128, 256)).float()

        return img_tensor, {'segmentation': label, 'depth': depth}

# =========================================================================
# 4. 评估逻辑
# =========================================================================
def evaluate(trainer, loader, device):
    trainer.model.eval()
    
    conf_mat = np.zeros((7, 7))
    depth_abs_err = 0.0
    depth_rel_err = 0.0
    depth_count = 0
    
    with torch.no_grad():
        for img, gts in tqdm(loader, leave=False):
            img = img.to(device)
            preds = trainer.model(img)
            preds = trainer.process_preds(preds)
            
            # --- Segmentation ---
            s_pred = preds['segmentation'].argmax(1).cpu().numpy()
            s_gt = gts['segmentation'].numpy()
            
            mask = (s_gt >= 0) & (s_gt < 7)
            if mask.sum() > 0:
                conf_mat += np.bincount(7 * s_gt[mask].astype(int) + s_pred[mask], minlength=49).reshape(7, 7)
            
            # --- Depth ---
            d_pred = preds['depth'].squeeze(1)     
            d_gt = gts['depth'].to(device)         
            
            if d_gt.shape != d_pred.shape:
                d_gt = d_gt.view_as(d_pred)
            
            valid = d_gt > 0
            if valid.sum() > 0:
                diff = torch.abs(d_pred[valid] - d_gt[valid])
                depth_abs_err += diff.sum().item()
                depth_rel_err += (diff / d_gt[valid]).sum().item()
                depth_count += valid.sum().item()

    # Metrics
    intersection = np.diag(conf_mat)
    union = conf_mat.sum(1) + conf_mat.sum(0) - intersection
    miou = np.nanmean(intersection / (union + 1e-10))
    pix_acc = intersection.sum() / (conf_mat.sum() + 1e-10)
    abs_err = depth_abs_err / (depth_count + 1e-10)
    rel_err = depth_rel_err / (depth_count + 1e-10)

    return miou, pix_acc, abs_err, rel_err

# =========================================================================
# 5. 主程序 (含 Weighting 自动解析)
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--cc_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--bs', '--batch_size', type=int, default=32, dest='bs')
    parser.add_argument('--output_txt', type=str, default='eval_final_report.txt')
    args = parser.parse_args()

    set_device(str(args.gpu_id))
    device = torch.device(f'cuda:{args.gpu_id}')
    
    if not os.path.exists(os.path.join(args.gt_dir, 'val', 'label')):
        print(f"❌ Error: Cannot find {args.gt_dir}/val/label/")
        return

    methods = sorted([d for d in os.listdir(args.checkpoint_dir) if os.path.isdir(os.path.join(args.checkpoint_dir, d))])
    corruptions = sorted([d for d in os.listdir(args.cc_dir) if os.path.isdir(os.path.join(args.cc_dir, d))])

    f_log = open(args.output_txt, 'w')
    def log(msg): print(msg); f_log.write(msg + '\n'); f_log.flush()
    
    log(f"🚀 Start Eval | Output=(128,256)")
    log(f"Metrics: Seg mIoU (↑) | Seg Pix Acc (↑) | Depth Abs Err (↓) | Depth Rel Err (↓)")

    # 定义支持的 Weighting 列表，用于从文件名中匹配
    SUPPORTED_WEIGHTINGS = ['EW', 'GradNorm', 'MGDA', 'UW', 'DWA', 'GLS', 'GradDrop', 
                            'PCGrad', 'GradVac', 'CAGrad', 'IMTL', 'RLW', 'MoCo', 'MoDo', 
                            'FAMO', 'FairGrad']

    for method in methods:
        log(f"\n{'='*10} {method} {'='*10}")
        ckpt_path = os.path.join(args.checkpoint_dir, method, 'best.pt')
        if not os.path.exists(ckpt_path): continue

        # [修正] 自动解析 Weighting
        weighting_method = 'EW' # 默认
        parts = method.split('_') # 假设命名如 cityscapes_GradNorm_HPS
        
        # 匹配 parts 中的关键词
        for w in SUPPORTED_WEIGHTINGS:
            # 必须精确匹配，防止 'Grad' 匹配到 'GradNorm' 
            if w in parts:
                weighting_method = w
                break
        
        log(f"   -> Detected Weighting: {weighting_method}")

        # 解析 Architecture
        arch = 'MTAN' if 'MTAN' in method.upper() else 'HPS'

        task_dict = {
            'segmentation': {'metrics':['mIoU', 'pixAcc'], 'metrics_fn':[], 'loss_fn':[], 'weight':[1]}, 
            'depth': {'metrics':['abs_err', 'rel_err'], 'metrics_fn':[], 'loss_fn':[], 'weight':[1]}
        }
        decoders = nn.ModuleDict({'segmentation': DeepLabHead(2048, 7), 'depth': DeepLabHead(2048, 1)})
        
        try:
            # 使用解析出的 weighting_method 初始化
            trainer = Citytrainer(
                task_dict=task_dict, 
                weighting=weighting_method, 
                architecture=arch,
                encoder_class=lambda: resnet_dilated('resnet50'), decoders=decoders,
                rep_grad=False, multi_input=False, optim_param={'optim':'adam'}, 
                scheduler_param=None, save_path=None, load_path=None, weight_args={}, arch_args={}
            )
            # 加载参数
            trainer.model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            trainer.model.eval()
        except Exception as e:
            log(f"❌ Init failed for {method}: {e}"); continue

        for corruption in corruptions:
            log(f"\n[Corruption: {corruption}]")
            metrics_sum = {'mIoU': 0, 'PixAcc': 0, 'AbsErr': 0, 'RelErr': 0}
            
            for severity in range(1, 6):
                dataset = CityscapesC_Dataset(
                    os.path.join(args.cc_dir, corruption, str(severity)), 
                    args.gt_dir
                )
                if len(dataset) == 0: continue
                
                loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)
                
                miou, pix_acc, abs_err, rel_err = evaluate(trainer, loader, device)
                
                log(f"  Level {severity}: mIoU={miou:.4f} | PixAcc={pix_acc:.4f} | AbsErr={abs_err:.4f} | RelErr={rel_err:.4f}")
                
                metrics_sum['mIoU'] += miou
                metrics_sum['PixAcc'] += pix_acc
                metrics_sum['AbsErr'] += abs_err
                metrics_sum['RelErr'] += rel_err
            
            log(f"  >> Avg: mIoU={metrics_sum['mIoU']/5:.4f} | PixAcc={metrics_sum['PixAcc']/5:.4f} | AbsErr={metrics_sum['AbsErr']/5:.4f} | RelErr={metrics_sum['RelErr']/5:.4f}")

    f_log.close()

if __name__ == "__main__":
    main()