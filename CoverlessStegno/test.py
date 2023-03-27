import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

opt = TestOptions().parse()
   
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]  

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')


for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    if i * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    
    out = model(data_i, mode='inference')
    if opt.save_per_img:
        root = save_root + ''
        if not os.path.exists(root + opt.name):
            os.makedirs(root + opt.name)
        imgs = out['fake_image'].data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for j in range(imgs.shape[0]):
                vutils.save_image(imgs[j:j+1], root + opt.name + '/' + str(i) + "-" + str(j) + "stego.png",
                        nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)

        label = masktorgb(data_i['label'].cpu().numpy())
        label = torch.from_numpy(label).float() / 128 - 1
        imgs = label.cpu()
        try:
            imgs = (imgs + 1) / 2
            for j in range(imgs.shape[0]):
                vutils.save_image(imgs[j:j + 1], root + opt.name + '/' + str(i) + "-" + str(j) + "seg.png",
                                  nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)

        imgs = data_i['ref'].cpu()
        try:
            imgs = (imgs + 1) / 2
            for j in range(imgs.shape[0]):
                vutils.save_image(imgs[j:j + 1], root + opt.name + '/' + str(i) + "-" + str(j) + "ref.png",
                                  nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)

        imgs = out['reveal_image'].data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for j in range(imgs.shape[0]):
                vutils.save_image(imgs[j:j + 1], root + opt.name + '/' + str(i) + "-" + str(j) + "reveal.png",
                                  nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)

        imgs = data_i['image_secret'].cpu()
        try:
            imgs = (imgs + 1) / 2
            for j in range(imgs.shape[0]):
                vutils.save_image(imgs[j:j + 1], root + opt.name + '/' + str(i) + "-" + str(j) + "secret.png",
                                  nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)
    else:
        if not os.path.exists(save_root + '/test/' + opt.name):
            os.makedirs(save_root + '/test/' + opt.name)

        label = masktorgb(data_i['label'].cpu().numpy())
        label = torch.from_numpy(label).float() / 128 - 1

        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu(),out['reveal_image'].data.cpu(),data_i['image_secret'].cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_root + '/test/' + opt.name + '/' + str(i) + '.png',  
                    nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)
