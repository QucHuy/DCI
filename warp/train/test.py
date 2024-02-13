import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
# from tensorboardX import SummaryWriter
import cv2
import datetime
import wandb
from util.util import train_log


opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)


def CreateDataset(opt):
    from data.cp_dataset import CPDataset
    dataset = CPDataset(opt.dataroot, mode='train', image_size=256)
    # print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


torch.distributed.init_process_group(backend="nccl")

os.makedirs('sample', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(0)
# local_rank = 0
device = torch.device(f'cuda:{0}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

warp_model = AFWM(opt, 3 + opt.label_nc)
print(warp_model)
warp_model.train()
warp_model.cuda()
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

model = warp_model

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in model.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.continue_train and opt.PBAFN_warp_checkpoint:
    checkpoint = torch.load(opt.PBAFN_warp_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_warp.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

# if local_rank == 0:
#     writer = SummaryWriter(path)

example_ct = 0
wandb.init(project="DCI-WARP")
wandb.config = {"learning_rate": opt.lr, "epochs": opt.niter + opt.niter_decay, "batch_size": opt.batchSize, "dataset" :"VTON HD"}
wandb.watch(model)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True
        example_ct += 1
        # input1
        c_paired = data['cloth']['paired'].cuda()
        cm_paired = data['cloth_mask']['paired']
        cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(np.float32)).cuda()
        # input2
        parse_agnostic = data['parse_agnostic'].cuda()
        densepose = data['densepose'].cuda()
        openpose = data['pose'].cuda()
        # GT
        label_onehot = data['parse_onehot'].cuda()  # CE
        label = data['parse'].cuda()  # GAN loss
        parse_cloth_mask = data['pcm'].cuda()  # L1
        im_c = data['parse_cloth'].cuda()  # VGG
        # visualization
        im = data['image']
        agnostic = data['agnostic']

        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)
        
        flow_out = model(input2, c_paired, cm_paired)
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        
