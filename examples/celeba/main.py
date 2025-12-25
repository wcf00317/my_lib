import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from create_dataset import celeba_dataloader, CELEBA_ATTRS

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

def parse_args(parser):
    # 添加 Scalability 实验所需的参数
    parser.add_argument('--num_tasks', default=40, type=int, help='number of tasks (attributes) to train (e.g. 5, 10, 20, 40)')
    parser.add_argument('--dataset_path', default='/data/chengfengwu/alrl/mtl_dataset/celeba/', type=str, help='dataset path')
    
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # 1. 动态定义任务列表
    if params.num_tasks < 1 or params.num_tasks > 40:
        raise ValueError('num_tasks must be between 1 and 40')
    
    current_tasks = CELEBA_ATTRS[:params.num_tasks]
    print(f"Running Experiment with {len(current_tasks)} tasks: {current_tasks}")

    # 2. 定义 task_dict
    # 每个属性是一个二分类任务，使用 AccMetric 和 CELoss
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]} for task in current_tasks}
    
    # 3. 准备 DataLoaders
    # 注意：CelebA 是 Multi-Label (Shared Input)，所以我们只需要传一个 DataLoader 即可
    # Trainer 会自动处理 yielded 的 label_dict
    data_loaders, _ = celeba_dataloader(batchsize=params.bs, 
                                        root_path=params.dataset_path,
                                        num_tasks=params.num_tasks)
    
    # LibMTL 对于 shared-input (如 NYU/CelebA) 通常直接接受 DataLoader 对象
    # 如果代码报错，可能需要将其包装为 {task: loader}，但在 Multi-Label 场景下通常不需要。
    # 这里我们按照 NYU 的风格，直接传递 loader。
    train_dataloaders = data_loaders['train']
    val_dataloaders = data_loaders['val']
    test_dataloaders = data_loaders['test']
    
    # 4. 定义 Encoder (ResNet-18)
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            hidden_dim = 512
            self.resnet_network = resnet18(pretrained=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = self.resnet_network(inputs)
            out = torch.flatten(self.avgpool(out), 1)
            out = self.hidden_layer(out)
            return out

    # 5. 定义 Decoders
    # 为每个任务定义一个分类头 (512 -> 2 classes)
    decoders = nn.ModuleDict({task: nn.Linear(512, 2) for task in current_tasks})
    
    # 6. 初始化 Trainer
    celebaModel = Trainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=Encoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input, # 默认为 False，适合 CelebA
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    
    if params.mode == 'train':
        celebaModel.train(train_dataloaders=train_dataloaders, 
                          val_dataloaders=val_dataloaders,
                          test_dataloaders=test_dataloaders, 
                          epochs=params.epochs)
    elif params.mode == 'test':
        celebaModel.test(test_dataloaders)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
