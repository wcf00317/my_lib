from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# CelebA 40 个属性的官方定义顺序
CELEBA_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

class CelebADataset(Dataset):
    def __init__(self, root_path, mode, num_tasks):
        self.root_path = root_path
        self.mode = mode
        self.num_tasks = num_tasks
        
        # 确定图片文件夹名称：优先使用用户指定的 'img_align_data'，如果不存在则回退到官方默认 'img_align_celeba'
        if os.path.exists(os.path.join(root_path, 'img_align_data')):
            self.img_folder = 'img_align_data'
        elif os.path.exists(os.path.join(root_path, 'img_align_celeba')):
            self.img_folder = 'img_align_celeba'
        else:
            # 如果都找不到，报错提示
            raise FileNotFoundError(f"Cannot find image folder 'img_align_data' or 'img_align_celeba' in {root_path}")

        # ResNet 标准预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 1. 读取 Partition 文件 (划分 Train/Val/Test)
        partition_path = os.path.join(root_path, 'list_eval_partition.txt')
        if not os.path.exists(partition_path):
             raise FileNotFoundError(f"Partition file not found: {partition_path}")
             
        with open(partition_path, 'r') as f:
            lines = f.readlines()
        
        # 0: Train, 1: Val, 2: Test
        target_split = {'train': '0', 'val': '1', 'test': '2'}[mode]
        self.img_names = []
        for line in lines:
            fname, split = line.strip().split()
            if split == target_split:
                self.img_names.append(fname)
        
        # 2. 读取 Attributes 文件
        attr_path = os.path.join(root_path, 'list_attr_celeba.txt')
        if not os.path.exists(attr_path):
             raise FileNotFoundError(f"Attribute file not found: {attr_path}")
             
        with open(attr_path, 'r') as f:
            lines = f.readlines()
        
        self.attr_dict = {}
        # 跳过前两行 (Count 和 Header)
        for line in lines[2:]:
            parts = line.strip().split()
            fname = parts[0]
            # 读取所有属性 (LibMTL 通常用 CELoss 做分类，所以这里我们把 -1 映射为 0, 1 映射为 1)
            # 原始数据: 1 (Present), -1 (Absent)
            labels = [1 if v == '1' else 0 for v in parts[1:]]
            self.attr_dict[fname] = labels

    def __getitem__(self, i):
        fname = self.img_names[i]
        img_path = os.path.join(self.root_path, self.img_folder, fname)
        
        image = Image.open(img_path).convert('RGB')
        
        all_labels = self.attr_dict[fname]
        # === 核心逻辑：只截取前 num_tasks 个任务 ===
        target_labels = all_labels[:self.num_tasks]
        
        # 构建返回字典
        label_dict = {}
        for idx in range(self.num_tasks):
            task_name = CELEBA_ATTRS[idx]
            # 转换为 LongTensor 以适配 CELoss (CrossEntropyLoss)
            label_dict[task_name] = torch.tensor(target_labels[idx]).long()
            
        return self.transform(image), label_dict

    def __len__(self):
        return len(self.img_names)

def celeba_dataloader(batchsize, root_path, num_tasks):
    # 为 Train/Val/Test 分别创建 DataLoader
    data_loader = {}
    for mode in ['train', 'val', 'test']:
        shuffle = True if mode == 'train' else False
        drop_last = True if mode == 'train' else False
        
        dataset = CelebADataset(root_path, mode, num_tasks)
        
        data_loader[mode] = DataLoader(dataset, 
                                       num_workers=4, 
                                       pin_memory=True, 
                                       batch_size=batchsize, 
                                       shuffle=shuffle,
                                       drop_last=drop_last)
    return data_loader, None # CelebA 不需要 iter_data_loader 这种复杂结构
