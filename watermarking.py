import os
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,help='which mode you want to choose, train, test or generate')
parser.add_argument('--gpu', type=str,help='gpu id to train')
parser.add_argument('--root-path', type=str, help='path of dataset')
parser.add_argument('--verbose', action='store_true', help='get tuned image')
parser.add_argument('--pretrain-watermarking', type=str, help='path of pretrained watermarking checkpoints')
parser.add_argument('--pretrain-detector', type=str, help='path of pretrained detector checkpoints')
parser.add_argument('--save-name', required=True, help='dir of checkpoints to save')
parser.add_argument('--save-image-path', default='./results', help='save compared image if verbose')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to train')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='number of batch size to train, min:4')
parser.add_argument('--mask-name', type=str, help='name of saved tuned checkpoints')
parser.add_argument('--save-mask-path', type=str, help='path of mask checkpoints')
parser.add_argument('--save-ckpt-path', type=str, help='path of mask checkpoints')
parser.add_argument('--output-json-path', type=str, help='path for detect watermarking result, just useful for test mode')
parser.add_argument('--finger-dim', type=int, default=64,help='lenth of watermarking')
# This is not necessary for all deepfake models.
parser.add_argument('--model_config', type=str, default='./hififace/config/model.yaml')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ["XDG_CACHE_HOME"] = "/data1/bh/pytorch_cache"




import numpy as np
import torch
import torch.nn as nn

import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import json

from base_network import get_mask_conv, get_fingerprint, dis_xcep, VGG19Loss
import dataset_utils

from noise_layers.noiser import Noiser
from noise_argparse import get_noise_layers

import bchlib
from encoding_utils import decode_BCH_fin,hamming_distance


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


def load_model_state(model,state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('%s: All params loaded' % type(model).__name__)
    else:
        print('%s: Some params were not loaded:' % type(model).__name__)
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def total_variation_loss(img):
    """
    Compute the Total Variation Loss.
    :param img: Tensor of the image. Shape (batch_size, channels, height, width)
    :return: Total Variation Loss.
    """
    # Calculate the difference of neighboring pixel-values
    # The variation is calculated for both the right and bottom neighboring pixels
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]  # Difference in height
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]  # Difference in width

    # Sum for all pixels and average over batch
    tv_loss = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
    tv_loss /= img.shape[0]
    return tv_loss


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


def get_fin_model(network):

    count = 0
    for i, (name, layer) in enumerate(reversed(list(network.generator.named_modules()))):
        # Traverse to query convolutional layers, in Hififace, the last module is "sff_module.f_up"
        if isinstance(layer, nn.Conv2d) and 'sff_module.f_up' in name :
            count += 1
            #count is the number of the total layer to check.
            if count<13:
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



    count = 0
    network.generator = load_model_state(network.generator, torch.load(os.path.join(args.save_mask_path, args.mask_name + '_M.pth')))



    for p in network.parameters():
        p.requires_grad = False

    for i,(name,layer) in enumerate(reversed(list(network.generator.named_modules()))):

        if isinstance(layer,get_mask_conv):
            name_list = name.split('.')
            temp_layer = network.generator
            for i,name_part in enumerate(name_list):
                if i == len(name_list)-1:
                    if name_part.isdigit():
                        temp_layer[int(name_part)] = get_fingerprint(temp_layer[int(name_part)],finger_dim)
                    else:
                        temp_layer = setattr(temp_layer, name_part,get_fingerprint(getattr(temp_layer,name_part),finger_dim))
                else:
                    if name_part.isdigit() :
                        temp_layer = temp_layer[int(name_part)]
                    else:
                        temp_layer = getattr(temp_layer,name_part)
                # count+=1




    for i, (name, layer) in enumerate(reversed(list(network.generator.named_modules()))):
        if isinstance(layer, get_fingerprint):
            layer.w_mask[layer.w_mask<3]=0
            layer.w_mask[layer.w_mask>=3]=6
            layer.set_mask()
            print("watermarking kernel size and embdding rate:")
            print(layer.ori_weight.size())
            print(layer.get_rate())
            # print(name)




    network = network.cuda()
    network.eval()

    #
    for name, layer in network.named_modules():
        if isinstance(layer, get_fingerprint):
            layer.register_forward_pre_hook(hook_fn)
            # print(name)
    #
    f_dis_image = dis_xcep(dim=finger_dim)
    f_dis_image = f_dis_image.cuda()
    f_dis_image.eval()


    noise_layers = get_noise_layers(0.5,1,0.5,1)
    enhance_image = Noiser(noise_layers=noise_layers,device='cuda:0')
    return network,f_dis_image,enhance_image


def train_fin_model():
    network, f_dis_image, enhance_image = get_fin_model(model)
    if args.pretrain_watermarking is not None:
        network.generator = load_model_state(network.generator, 
                                             torch.load(args.pretrain_watermarking))
    if args.pretrain_detector is not None:
        f_dis_image = load_model_state(f_dis_image,
                                   torch.load(args.pretrain_detector))
        f_dis_image.eval()



    f_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.generator.parameters()),
                                   betas=(0.5, 0.999), lr=1e-5,
                                   weight_decay=1e-4)

    d_schelduler = torch.optim.lr_scheduler.CosineAnnealingLR(f_optimizer, T_max=10, eta_min=1e-6)

    VGG_loss = VGG19Loss('cuda:0')
    for epoch in range(args.epochs):
        with tqdm(train_loader, total=len(train_loader), ncols=0) as pbar:
            for index, (image_face,arc_face) in enumerate(pbar):
                arc_face = arc_face.cuda()
                image_face = image_face.cuda()
                fingers = torch.randint(0, 2, (args.batch_size, finger_dim), dtype=torch.float).cuda()

                global fingerprint
                fingerprint = fingers

                ############## Forward Pass ######################
                # please follow the target deepfake model
                with torch.no_grad():
                    ori_img_fake, ori_i_low, ori_m_r, ori_m_low = ori_model.generator.interp(image_face, arc_face, arc_face, rate, mode)
                new_image_fake, new_i_low, new_m_r, new_m_low = network.generator.interp(image_face, arc_face, arc_face, rate, mode)
                ##################################################



                v_loss = VGG_loss(new_image_fake, ori_img_fake.detach())
                g_loss = mse_loss_fn(new_image_fake, ori_img_fake.detach())+mse_loss_fn(new_m_r, ori_m_r.detach())

                en_image = enhance_image([new_image_fake,ori_img_fake])
                f_image_printor = f_dis_image(en_image[0])
                b_loss = bce_loss_fn(f_image_printor, fingers)
                tv_loss = total_variation_loss(new_image_fake)
                loss = 100*g_loss+10*b_loss+v_loss+0.0001*tv_loss

                f_optimizer.zero_grad()
                loss.backward()
                f_optimizer.step()

                if args.verbose:
                    if index % 300 == 0:
                        source_image2 = arc_face[0:4, :, :, :]
                        gen_image = new_image_fake[0:4, :, :, :]
                        ori_gen_image = ori_img_fake[0:4, :, :, :]
                        diff = abs(gen_image-ori_gen_image)*10
                        gen = torch.vstack((diff, source_image2,gen_image,ori_gen_image))
                        save_image(gen, os.path.join(args.save_image_path, '{}_.png'.format(index)))


                pbar.set_postfix(
                    { 'g_loss': g_loss.item(),'b_loss': b_loss.item(),'v_loss':v_loss.item(),'tv_loss':tv_loss.item()})
            d_schelduler.step()

        torch.save(network.generator.state_dict(), os.path.join(args.save_ckpt_path, args.save_name + '_MF.pth'))
        torch.save(f_dis_image.state_dict(),
                   os.path.join(args.save_ckpt_path, args.save_name + '_dis_image.pth'))
        


def test_watermark():
    act_res = list()
    res_dict = dict()
    act = nn.Sigmoid().cuda()
    path_list = list()
    BCH_POLYNOMIAL = 131
    BCH_BITS = 1
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)



    test_dataset = dataset_utils.dataset_for_detect(data_root=args.root_path)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=32,
                                               pin_memory=True,
                                               shuffle=True,
                                               sampler=None,
                                               drop_last=True)
    network, f_dis_image, enhance_image = get_fin_model(model)


    f_dis_image = load_model_state(f_dis_image,torch.load(args.pretrain_detector))
    f_dis_image = f_dis_image.eval()

    # you can change the 63-bit watermark
    source_fingerprint = '110101000000001011111110101001101011011111011010111010100110101'


    

    with tqdm(test_loader, total=len(test_loader), ncols=0) as pbar:
        for index, (image_face,image_path) in enumerate(pbar):
    #
            image_face = image_face.cuda()
            fingers = torch.randint(0, 2, (args.batch_size, finger_dim), dtype=torch.float).cuda()

            global fingerprint
            fingerprint = fingers
    #
            with torch.no_grad():
                finger_res = f_dis_image(image_face)
                act_res.extend(act(finger_res).cpu().numpy())
            path_list.extend(image_path)

    act_res = np.array(act_res)
    act_res[act_res < 0.5] = 0
    act_res[act_res >= 0.5] = 1
    for i, hamming_res in enumerate(act_res):
        hamming_res = hamming_res.astype(np.uint8)
        list_of_ints = hamming_res.tolist()
        binary_str = ''.join(str(i) for i in list_of_ints)
        res, flag = decode_BCH_fin(64, binary_str, bch)
    
        if flag.count(-1) <= 3:
            distance = hamming_distance(binary_str, source_fingerprint)
            if distance <= 16: #defined in our paper
                res_dict.update({path_list[i]:1}) # 1 means the watermark is matched
            else:
                res_dict.update({path_list[i]:0})
        else:
            res_dict.update({path_list[i]:0})               
    with open(args.output_json_path, 'w') as f:
        json.dump(res_dict, f, indent=4)



def generate_method():

    # for the img_size please refer to the Deepfake model
    test_dataset = dataset_utils.Hififace_ClassifierDataset(data_root=args.root_path,img_size=256)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=32,
                                               pin_memory=True,
                                               shuffle=True,
                                               sampler=None,
                                               drop_last=True)
    network, f_dis_image, enhance_image = get_fin_model(model)
    network.generator = load_model_state(network.generator, 
                                    torch.load(args.pretrain_watermarking))    
    network = network.eval()

    # 63 bch bit watermark
    source_fingerprint = '110101000000001011111110101001101011011111011010111010100110101'


    count=0
    with tqdm(test_loader, total=len(test_loader), ncols=0) as pbar:
        for index, (image_face,arc_face) in enumerate(pbar):
    #
            arc_face = arc_face.cuda()
            image_face = image_face.cuda()

            hamming_fingers = torch.zeros(args.batch_size, finger_dim).cuda()
            b_num = list()
            for i in range(args.batch_size):

                bch_fingers = source_fingerprint
                int_list = [int(b) for b in bch_fingers]
                #add 0 to 64 bit
                int_list.append(0)
                b_num.append(count)
                count+=1
                hamming_fingers[i] = torch.tensor(int_list).cuda()

            bch_fingers = hamming_fingers.float()

            global fingerprint
            fingerprint = bch_fingers

            ############## Forward Pass ######################
            # please follow the target deepfake model

            with torch.no_grad():
                new_image_fake, new_i_low, new_m_r, new_m_low = network.generator.interp(image_face, arc_face,
                                                                                             arc_face, rate, mode)
                ori_img_fake, ori_i_low, ori_m_r, ori_m_low = ori_model.generator.interp(image_face, arc_face, arc_face,
                                                                                         rate, mode)
            ##################################################

            for i in range(len(ori_img_fake)):
                save_image(ori_img_fake[i].unsqueeze(0), os.path.join(args.save_image_path, 'original', '{}_{}.png'.format(index, i)))
                save_image(new_image_fake[i].unsqueeze(0),
                           os.path.join(args.save_image_path, 'watermarked', '{}_{}.png'.format(index, i)))





if __name__ == '__main__':
    if os.path.exists(args.save_image_path):
        # Clear the folder
        shutil.rmtree(args.save_image_path)
        os.makedirs(args.save_image_path)
    else:
        # Create the folder
        os.makedirs(args.save_image_path)

    if args.mode == 'generate':
        os.makedirs(os.path.join(args.save_image_path,'watermarked'))
        os.makedirs(os.path.join(args.save_image_path,'original'))


    if args.mode == 'train':
        train_fin_model()
    elif args.mode == 'generate':
        generate_method()
    elif args.mode == 'test':
        test_watermark()
