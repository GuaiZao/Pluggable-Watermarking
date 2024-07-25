import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,help='gpu id to train')
parser.add_argument('--root-path', type=str, help='path of dataset')
parser.add_argument('--verbose', action='store_true', help='get tuned image')
parser.add_argument('--save-path', default='./results', help='save compared image if verbose')
parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to train')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='number of batch size to train, min:4')
parser.add_argument('--mask-name', type=str, help='name of saved tuned checkpoints')
parser.add_argument('--save-mask-path', default='False', help='path of mask checkpoints')
parser.add_argument('--finger-dim', type=int, default=64,help='lenth of watermarking')
# This is not necessary for all deepfake models.
parser.add_argument('--model-config', type=str, default='./hififace/config/model.yaml')
# parser.add_argument('--model_checkpoint_path', type=str, required=True)


args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ["XDG_CACHE_HOME"] = "/data1/bh/pytorch_cache"



import torch
import torch.nn as nn

import torchvision
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm

from base_network import get_mask_conv, VGG19Loss
import dataset_utils
import numpy as np




# add path of target model to import,for example:
sys.path.append('./hififace')
sys.path.append('./hififace/model/Deep3DFaceRecon_pytorch')

# import target deepfake model and layers, for example:
from hififace.hififace_pl import HifiFace
from omegaconf import OmegaConf

# target deepfake model path and other paramerters, for example:
model_checkpoint_path = './hififace/hififace_opensouce_299999.ckpt'
mode = 'all'
rate = 0.5


finger_dim = args.finger_dim
fingerprint = None

def hook_fn(module, args):
    return [args[0],fingerprint]

# define target deepfake model twice, for example:
# first
model = HifiFace(OmegaConf.load(args.model_config))
checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()
for p in model.parameters():
    p.requires_grad = False

# second
ori_model = HifiFace(OmegaConf.load(args.model_config))
checkpoint = torch.load(model_checkpoint_path)
ori_model.load_state_dict(checkpoint['state_dict'])
ori_model.eval()
ori_model.cuda()
for p in ori_model.parameters():
    p.requires_grad = False

# loss function
mse_loss_fn = nn.MSELoss()
bce_loss_fn = nn.BCEWithLogitsLoss()

# image size is depended on the target deepfake model, for example, hififace is 256X256
re256 = torchvision.transforms.Resize(256)

# dataset loader, Please adjust according to the actual situation to meet the input requirements of the target deepfake model, for example for hififace:
train_dataset = dataset_utils.Hififace_ClassifierDataset(data_root=args.root_path,img_size=256)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=32,
                                           pin_memory=True,
                                           shuffle=True,
                                           sampler=None,
                                           drop_last=True)

def get_fingerprint_local(network):
    for p in network.parameters():
        p.requires_grad = False
    print("some layers need to determine the embedding position:")
    count=0
    for name, layer in reversed(list(network.generator.named_modules())):
        # Traverse to query convolutional layers, in Hififace, the last module is "sff_module.f_up"
        if isinstance(layer, nn.Conv2d) and 'sff_module.f_up' in name:
            count+=1
            #count is the number of the total layer to check.
            if count<13:
                print(name)
                name_list = name.split('.')
                temp_layer = network.generator
                for i, name_part in enumerate(name_list):
                    if i == len(name_list) - 1:
                        if name_part.isdigit():
                            temp_layer[int(name_part)] = get_mask_conv(temp_layer[int(name_part)],'cuda:0')
                        else:
                            temp_layer = setattr(temp_layer, name_part,
                                                 get_mask_conv(getattr(temp_layer, name_part),'cuda:0'))
                    else:
                        if name_part.isdigit():
                            temp_layer = temp_layer[int(name_part)]
                        else:
                            temp_layer = getattr(temp_layer, name_part)


    f_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,network.generator.parameters()), lr=1e-2,
                                   betas=(0.0, 0.99),
                                   weight_decay=1e-4)

    VGG_loss = VGG19Loss('cuda:0')

    for epoch in range(args.epochs):

        with tqdm(train_loader, total=len(train_loader), ncols=0) as pbar:
            for index, (image_face, arc_face) in enumerate(pbar):
                arc_face = arc_face.cuda()
                image_face = image_face.cuda()


                ############## Forward Pass ######################
                # please follow the target deepfake model
                with torch.no_grad():
                    ori_img_fake,ori_i_low, ori_m_r, ori_m_low = ori_model.generator.interp(image_face,arc_face,arc_face,rate,mode)

                new_image_fake,new_i_low, new_m_r, new_m_low = network.generator.interp(image_face,arc_face,arc_face,rate,mode)

                ##################################################
                weight_loss = 0
                bias_loss = 0
                for name,layer in network.generator.named_modules():
                    if isinstance(layer,get_mask_conv):
                        # if layer.weight_mask.grad is not None:
                        #     print(layer.weight_mask.grad.mean(),(layer.weight_mask.max()-layer.weight_mask.min()))
                        weight_loss += layer.get_weight_mask_loss(layer.weight_mask)
                        bias_loss += layer.get_weight_mask_loss(layer.bias_mask)


                v_loss = VGG_loss(new_image_fake,ori_img_fake.detach())
                g_loss = mse_loss_fn(new_image_fake,ori_img_fake.detach())+mse_loss_fn(new_m_r,ori_m_r.detach())
                loss = weight_loss+bias_loss+10000*g_loss+100*v_loss
                f_optimizer.zero_grad()
                loss.backward()
                f_optimizer.step()

                if args.verbose:
                    if index % 300 == 0:
                        source_image = ori_img_fake[0:4, :, :, :]
                        gen_image = new_image_fake[0:4, :, :, :]
                        res = abs(gen_image-source_image)*10
                        gen = torch.vstack((source_image, gen_image,res))
                        save_image(gen, os.path.join(args.save_path, '{}_.png'.format(index)))


                pbar.set_postfix(
                    {'g_loss': g_loss.item(), 'wei_loss': weight_loss.item(), 'bias_loss': bias_loss.item(),'vgg':v_loss.item()})

        torch.save(network.generator.state_dict(), os.path.join(args.save_mask_path, args.mask_name + '_M.pth'))

if __name__ == '__main__':
    get_fingerprint_local(model)


