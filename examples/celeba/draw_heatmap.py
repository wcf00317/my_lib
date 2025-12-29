import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 引入 LibMTL 相关模块
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import prepare_args, LibMTL_args
from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from create_dataset import celeba_dataloader, CELEBA_ATTRS

# ==========================================
# 1. 重新定义 Encoder (必须与 main.py 中完全一致)
# ==========================================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        hidden_dim = 512
        self.resnet_network = resnet18(pretrained=False) # 加载权重时不需要预训练参数，因为会覆盖
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        
    def forward(self, inputs):
        out = self.resnet_network(inputs)
        out = torch.flatten(self.avgpool(out), 1)
        out = self.hidden_layer(out)
        return out

def compute_gradient_heatmap(params):
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 2. 准备配置和模型
    # ==========================================
    # 模拟参数配置 (你需要根据你实际训练的设置修改这里)
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    # 确定任务数量
    current_tasks = CELEBA_ATTRS[:params.num_tasks]
    print(f"Analyzing {len(current_tasks)} tasks...")

    # 定义 task_dict (仅用于初始化 Trainer)
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]} for task in current_tasks}
    
    # 定义 Decoders
    decoders = nn.ModuleDict({task: nn.Linear(512, 2) for task in current_tasks})
    
    # 初始化 Trainer
    # 注意：这里我们传入 load_path，Trainer 会自动加载权重
    trainer = Trainer(task_dict=task_dict, 
                      weighting=params.weighting, 
                      architecture=params.arch, 
                      encoder_class=Encoder, 
                      decoders=decoders,
                      rep_grad=params.rep_grad,
                      multi_input=params.multi_input,
                      optim_param=optim_param,
                      scheduler_param=scheduler_param,
                      save_path=None,
                      load_path=params.load_path, # 这里填你的 checkpoints 路径
                      **kwargs)
    
    # ==========================================
    # 3. 获取一个 Batch 的数据
    # ==========================================
    # 只需要 Test 或 Val 的数据即可
    data_loaders, _ = celeba_dataloader(batchsize=params.bs, 
                                        root_path=params.dataset_path,
                                        num_tasks=params.num_tasks)
    loader = data_loaders['test']
    
    # 获取一个 batch
    try:
        batch_data, batch_labels = next(iter(loader))
    except:
        # 如果 loader 是 list 形式 (LibMTL 内部处理)
        batch_data, batch_labels = next(iter(loader))

    # 移动到 GPU
    batch_data = batch_data.to(device)
    for task in current_tasks:
        batch_labels[task] = batch_labels[task].to(device)

    # ==========================================
    # 4. 计算梯度并构建相似度矩阵
    # ==========================================
    model = trainer.model
    model.eval() # 设为 eval 模式 (但在计算梯度前要 zero_grad)
    
    # 这里的 "Shared Encoder 最后一层" 指的是 Encoder 类中 hidden_layer 的第一个 Linear 层
    # 它的梯度最能代表 Shared Representation 的冲突情况
    # 路径: model -> encoder -> hidden_layer -> [0] (Linear)
    # 注意: 具体路径可能因 Architecture (HPS/AutoLambda等) 不同而略有差异
    # 对于默认的 HPS 架构，model 本身继承了 Encoder (通过 Mixin)，或者包含 encoder 属性
    # 查看 trainer.py 的 _prepare_model，它是组合类。
    # 最稳妥的方式是直接找 model.encoder (如果是 HPS，通常在 Trainer 初始化时被赋值)
    # 或者是 model (如果 model 就是 Encoder 的子类)。
    # 根据 trainer.py: self.model = MTLmodel(...)
    # 让我们假设是 HPS 架构，通常它有 encoder 属性。
    
    if hasattr(model, 'encoder'):
        shared_layer = model.encoder.hidden_layer[0] 
    else:
        # 如果是单层结构或者直接继承
        # 根据 CelebA 的 Encoder 定义，最后一层是 hidden_layer[0]
        # 我们尝试通过 named_modules 找到它
        shared_layer = model.hidden_layer[0]

    print(f"Hooking gradients on layer: {shared_layer}")

    gradients = {}
    
    # 清空梯度
    model.zero_grad()

    for task in current_tasks:
        # 1. Forward
        # 对于 HPS，forward 返回所有任务的预测字典
        preds = model(batch_data) 
        
        # 2. Compute Loss for single task
        # 注意：这里我们只计算单任务的 Loss
        loss = trainer.meter.losses[task]._update_loss(preds[task], batch_labels[task])
        
        # 3. Backward (Retain Graph is crucial!)
        # 我们只想要 Shared Layer 的梯度，不要更新参数
        # autograd.grad 比 backward() 更干净，因为它直接返回梯度而不累积到 .grad 属性
        grad = torch.autograd.grad(loss, shared_layer.parameters(), retain_graph=True)
        
        # grad 是一个 tuple (weight_grad, bias_grad)，我们需要展平并拼接成一个向量
        grad_vec = torch.cat([g.view(-1) for g in grad])
        gradients[task] = grad_vec
        
        # 清理图以节省显存 (可选，但 retain_graph=True 可能会占内存)
        # model.zero_grad() # autograd.grad 不影响 .grad，所以其实不需要 zero_grad

    # ==========================================
    # 5. 计算余弦相似度矩阵
    # ==========================================
    num_tasks = len(current_tasks)
    heatmap_matrix = np.zeros((num_tasks, num_tasks))
    
    for i, t1 in enumerate(current_tasks):
        for j, t2 in enumerate(current_tasks):
            g1 = gradients[t1]
            g2 = gradients[t2]
            
            # Cosine Similarity Formula
            # sim = (g1 . g2) / (|g1| * |g2|)
            sim = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2) + 1e-8)
            heatmap_matrix[i, j] = sim.item()

    # ==========================================
    # 6. 绘图
    # ==========================================
    plt.figure(figsize=(20, 18))
    sns.set(font_scale=0.8)
    
    # 绘制热力图
    # vmin=-1, vmax=1 确保颜色映射是对称的 (蓝色-1, 白色0, 红色1)
    # cmap='RdBu_r' : Red (Positive), White (Zero), Blue (Negative) - 适合展示冲突(负相关)
    # 或者用 'coolwarm'
    sns.heatmap(heatmap_matrix, 
                xticklabels=current_tasks, 
                yticklabels=current_tasks,
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={"shrink": .5})
    
    plt.title(f"Gradient Conflict Heatmap ({params.weighting})", fontsize=20)
    plt.tight_layout()
    
    save_file = f'heatmap_{params.weighting}.png'
    plt.savefig(save_file, dpi=300)
    print(f"Heatmap saved to {save_file}")

if __name__ == "__main__":
    # ==========================================
    # 修改参数解析逻辑
    # ==========================================
    # 1. 获取 LibMTL 的基础参数解析器
    parser = LibMTL_args 
    
    # 2. 手动添加 celeba/main.py 中定义的特定参数
    parser.add_argument('--num_tasks', default=40, type=int, help='number of tasks (attributes) to train')
    parser.add_argument('--dataset_path', default='./data/celeba', type=str, help='dataset path')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    
    # 3. 解析参数
    params = parser.parse_args()
    
    # ==========================================
    # 参数检查与运行
    # ==========================================
    if params.load_path is None:
        print("Warning: No load_path provided. Running with random weights (Results will be meaningless).")
    
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    
    # 运行主函数
    compute_gradient_heatmap(params)