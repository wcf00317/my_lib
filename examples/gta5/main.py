import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from PIL import Image
import glob
import fnmatch
import random
import argparse
from torchvision import transforms

# LibMTL imports
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.trainer import Trainer
from LibMTL.utils import set_random_seed
from LibMTL.model import resnet_dilated
# 必须导入 AbsLoss，否则自定义 Loss 无法被 meter 记录
from LibMTL.loss import AbsLoss

# Metric 基类防报错
try:
    from LibMTL.metrics import AbsMetric
except ImportError:
    class AbsMetric(object): pass

# ============================================================================
# 0. ASPP Decoder Definitions (LibMTL Standard)
# ============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

# ============================================================================
# 1. Metric & Loss Definition (FIXED for LibMTL API)
# ============================================================================
class ConfMatrix(AbsMetric):
    def __init__(self, num_classes):
        super(ConfMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None

    # 【关键修复】方法名必须是 update_fun，参数名必须是 pred, gt
    def update_fun(self, pred, gt):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        
        with torch.no_grad():
            if pred.dim() == 4:
                pred = pred.argmax(1)
            pred = pred.flatten()
            gt = gt.flatten()
            k = (gt >= 0) & (gt < n)
            inds = n * gt[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    # 【关键修复】方法名必须是 score_fun，且必须返回列表 list
    def score_fun(self):
        if self.mat is None:
            return [0.0, 0.0]
        h = self.mat.float()
        # Pixel Accuracy
        acc_global = torch.diag(h).sum() / h.sum()
        # mIoU
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        miou = torch.nanmean(iu)
        # LibMTL 的 _record.py 要求返回 list，按 metrics 定义顺序
        return [miou.item(), acc_global.item()]

    def reinit(self):
        self.mat = None

class SegLoss(AbsLoss):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())

# ============================================================================
# 2. Dataset Definitions
# ============================================================================
GTA5_TO_7_CLASSES = {
    0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1,
    7: 0, 8: 0, 9: -1, 10: -1, 11: 1, 12: 1, 13: 1,
    14: -1, 15: -1, 16: -1, 17: 2, 18: -1, 19: 2, 20: 2,
    21: 3, 22: 3, 23: 4, 24: 5, 25: 5, 26: 6, 27: 6,
    28: 6, 29: -1, 30: -1, 31: 6, 32: 6, 33: 6
}

class GTA5Dataset(Dataset):
    def __init__(self, root_dir, img_size):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.images = []
        self.targets = []
        
        search_dirs = ['', 'train', 'val', 'Train', 'Val', 'gta5']
        found_any = False
        
        for subdir in search_dirs:
            img_dir_candidates = [
                os.path.join(root_dir, subdir, 'images'),
                os.path.join(root_dir, subdir, 'Images')
            ]
            lbl_dir_candidates = [
                os.path.join(root_dir, subdir, 'labels'),
                os.path.join(root_dir, subdir, 'Labels')
            ]
            
            img_dir = next((d for d in img_dir_candidates if os.path.isdir(d)), None)
            lbl_dir = next((d for d in lbl_dir_candidates if os.path.isdir(d)), None)
            
            if img_dir and lbl_dir:
                found_any = True
                files = glob.glob(os.path.join(img_dir, "*.png"))
                for img_path in files:
                    file_name = os.path.basename(img_path)
                    label_path = os.path.join(lbl_dir, file_name)
                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.targets.append(label_path)
        
        if not found_any:
             print(f"[GTA5 Error] Could not find 'images'/'labels' folders in {root_dir}")

        self.images.sort()
        self.targets.sort()
        self.mapping = np.zeros(256, dtype=np.int64) - 1
        for k, v in GTA5_TO_7_CLASSES.items():
            if k >= 0: self.mapping[k] = v
            
        print(f"[GTA5] Found {len(self.images)} samples. Resize Target: {self.img_size}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.targets[idx])
        
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
            
        rgb_tensor = transforms.ToTensor()(img).float()
        label_np = np.array(label, dtype=np.int64)
        label_np[label_np > 255] = 255
        label_mapped = self.mapping[label_np]
        seg_tensor = torch.from_numpy(label_mapped).long()
        return {'rgb': rgb_tensor, 'segmentation': seg_tensor}

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        super().__init__()
        self.root = os.path.expanduser(root_dir)
        self.split = split
        folder_split = 'train' if split == 'train' else 'val'
        self.data_path = os.path.join(self.root, folder_split)
        
        if not os.path.exists(self.data_path):
            if os.path.basename(root_dir.rstrip('/')) == folder_split:
                 self.data_path = root_dir
            else:
                 raise ValueError(f"Data path not found: {self.data_path}")
                 
        image_dir = os.path.join(self.data_path, 'image')
        if not os.path.exists(image_dir):
            raise ValueError(f"Image dir not found: {image_dir}")
        
        self.index_list = fnmatch.filter(os.listdir(image_dir), '*.npy')
        self.index_list = [int(x.replace('.npy', '')) for x in self.index_list]
        self.index_list.sort()
        self.num_samples = len(self.index_list)
        print(f"[Cityscapes {split.upper()}] Found {self.num_samples} samples")

    def __getitem__(self, i):
        index = self.index_list[i]
        img_np = np.load(os.path.join(self.data_path, 'image', f'{index}.npy'))
        label_np = np.load(os.path.join(self.data_path, 'label', f'{index}.npy'))
        if img_np.ndim == 3:
            image = torch.from_numpy(np.moveaxis(img_np, -1, 0)).float()
        else:
            image = torch.from_numpy(img_np).float().unsqueeze(0)
        semantic = torch.from_numpy(label_np).long()
        return {'rgb': image, 'segmentation': semantic}

    def __len__(self):
        return self.num_samples

class LibMTLDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data['rgb'], {'segmentation': data['segmentation']}

# ============================================================================
# 3. Main Function
# ============================================================================
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_random_seed(params.seed)

    # 1. Define Tasks
    task_dict = {
        'segmentation': {
            'metrics': ['mIoU', 'Acc'], 
            'metrics_fn': ConfMatrix(num_classes=7), # 使用修正后的 Metric
            'loss_fn': SegLoss(), # 使用修正后的 Loss
            'weight': [1.0]
        }
    }

    # 2. Define Network
    def encoder_class():
        return resnet_dilated('resnet50')
    decoders = nn.ModuleDict({'segmentation': DeepLabHead(2048, 7)})

    # 3. Data Auto-Align
    if params.cityscapes_path:
        cityscapes_root = params.cityscapes_path
    else:
        cityscapes_root = os.path.join(params.dataset_path, 'cityscapes_npy')

    try:
        probe_ds = CityscapesDataset(root_dir=cityscapes_root, split='val')
        if len(probe_ds) > 0:
            sample = probe_ds[0]
            real_h, real_w = sample['rgb'].shape[-2:]
            img_size = (real_h, real_w)
            print(f"\n[Auto-Align] Detected Cityscapes .npy shape: {img_size} (H, W)")
            test_ds = probe_ds
            test_wrapper = LibMTLDatasetWrapper(test_ds)
        else:
            raise ValueError("Cityscapes dataset empty")
    except Exception as e:
        print(f"[Warning] Cityscapes probing failed: {e}")
        print("[Fallback] Using (288, 384)")
        img_size = (288, 384)
        test_wrapper = None

    gta5_root = params.dataset_path
    gta5_ds = GTA5Dataset(root_dir=gta5_root, img_size=img_size)
    
    if len(gta5_ds) == 0:
        raise ValueError(f"GTA5 Dataset empty at {gta5_root}")

    train_len = int(0.95 * len(gta5_ds))
    val_len = len(gta5_ds) - train_len
    train_subset, val_subset = random_split(gta5_ds, [train_len, val_len], 
                                            generator=torch.Generator().manual_seed(params.seed))
    
    train_wrapper = LibMTLDatasetWrapper(train_subset)
    val_wrapper = LibMTLDatasetWrapper(val_subset)
    
    if test_wrapper is None:
        test_wrapper = val_wrapper

    train_loader = DataLoader(train_wrapper, batch_size=params.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_wrapper, batch_size=params.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_wrapper, batch_size=params.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("-" * 50)
    print(f"Source Train (GTA5): {len(train_wrapper)}")
    print(f"Source Val   (GTA5): {len(val_wrapper)}")
    print(f"Target Test  (City): {len(test_wrapper)}")
    print("-" * 50)

    class SegTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                     rep_grad, multi_input, optim_param, scheduler_param, target_size, **kwargs):
            super().__init__(task_dict=task_dict, 
                             weighting=weighting, 
                             architecture=architecture, 
                             encoder_class=encoder_class, 
                             decoders=decoders,
                             rep_grad=rep_grad,
                             multi_input=multi_input,
                             optim_param=optim_param, 
                             scheduler_param=scheduler_param,
                             **kwargs)
            self.target_size = target_size

        def process_preds(self, preds, task_name=None):
            if self.multi_input:
                return F.interpolate(preds, size=self.target_size, mode='bilinear', align_corners=True)
            else:
                for task in self.task_name:
                    preds[task] = F.interpolate(preds[task], size=self.target_size, mode='bilinear', align_corners=True)
                return preds

    trainer = SegTrainer(task_dict=task_dict,
                         weighting=params.weighting,
                         architecture=params.arch,
                         encoder_class=encoder_class,
                         decoders=decoders,
                         rep_grad=params.rep_grad,
                         multi_input=params.multi_input,
                         save_path=params.save_path,
                         load_path=params.load_path,
                         optim_param=optim_param,
                         scheduler_param=scheduler_param,
                         target_size=img_size,
                         **kwargs)

    # 4. Train
    trainer.train(train_loader,      
                  test_loader,       
                  params.epochs,     
                  val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = LibMTL_args
    parser.add_argument('--dataset_path', default='./data', type=str, help='Root path to gta5')
    parser.add_argument('--cityscapes_path', default=None, type=str, help='Root containing cityscapes val/image/*.npy')
    parser.add_argument('--bs', dest='batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    args = parser.parse_args()
    main(args)