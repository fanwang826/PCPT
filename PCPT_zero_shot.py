import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
import os.path as osp

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch import nn
import os
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

import copy
from common.utils.analysis import collect_feature, tsne
from common.utils.data import ForeverDataIterator

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, get_coop_oh
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.office_home_prompts import office_home_classes
from data.office_prompts import office_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


ID_to_DIRNAME={
    'Art': 'Art_ima',
    'Clipart': 'Cli_ima',
    'Product': 'Pro_ima',
    'Real_world': 'Rew_ima',
    'amazon':'ama_ima',
    'dslr':'dsl_ima',
    'webcam':'web_ima',
    'clipart':'clipart_img',
    'painting':'painting_img',
    'real':'real_img',
    'sketch':'sketch_img',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
        

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def data_load(args,data_root,set_id): 
    ## prepare data
    testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
    txt_test = datasets.ImageFolder(testdir)
    txt_test = txt_test.imgs

    new_tar = []
    for i in range(len(txt_test)):
            rec = txt_test[i]
            line = rec[0] + ' ' + str(int(rec[1])) + '\n'
            new_tar.append(line)
        
    txt_test = new_tar.copy()

    return txt_test, txt_test


def test_time_tuning(model, inputs, target, optimizer, scaler,args):


    with torch.cuda.amp.autocast():
        output, _, _ = model(inputs) 
        probs = output.softmax(dim=-1).detach().cpu().numpy()
        a = torch.tensor(probs)
        idx = probs.argmax(1)

        lbl_pseu = torch.zeros(len(output)).cuda() + torch.tensor(idx).cuda()
        #cross-entropy loss
        loss_c = F.cross_entropy(output, lbl_pseu.long())
        
        print(torch.sum(lbl_pseu== target)/len(lbl_pseu))
        
    optimizer.zero_grad()
    # compute gradient and do SGD step
    scaler.scale(loss_c).backward()
    # Unscales the gradients of optimizer's assigned params in-place
    scaler.step(optimizer)
    scaler.update()
    
    return 

def test_time_tuning_neighbor(model, inputs, target, optimizer, scaler,args,memo,memo_pro):

    #compute prototypes
    all_fea = np.array(memo)
    aff = np.array(memo_pro)
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
   
    with torch.cuda.amp.autocast():
        output, image_features, _ = model(inputs) 
        probs = output.softmax(dim=-1).detach().cpu().numpy()
        a = torch.tensor(probs)
        idx = probs.argmax(1)
        dd = cdist(image_features.detach().cpu().numpy(), initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        #assign pseudo-labels for samples based on class prototypes
        pred_label = torch.tensor(pred_label).cuda()
    
        
        lbl_pseu = torch.zeros(len(output)).cuda() + torch.tensor(idx).cuda()
        #cross-entropy loss  
        loss_c = F.cross_entropy(output, pred_label.long())
        softmax_out = nn.Softmax(dim=1)(output)
        entropy = -softmax_out  * torch.log(softmax_out + 1e-3)
        entropy = torch.sum(entropy, dim=1)
        msoftmax = softmax_out.mean(dim=0)

        #diverse loss
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-3))

        loss_c = loss_c - args.alpha * gentropy_loss
       
        print(torch.sum(pred_label== target)/len(lbl_pseu))
        
    optimizer.zero_grad()
    # compute gradient and do SGD step
    scaler.scale(loss_c).backward()
    # Unscales the gradients of optimizer's assigned params in-place
    scaler.step(optimizer)
    scaler.update()
    
    return 


def main(args):
    # args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))

    elif args.dset == 'office-home':
        classnames = office_home_classes
    elif args.dset == 'office':
        classnames = office_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop_oh(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        
        print("Use pre-trained soft prompt source model as initialization")
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            data_transform =  transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
            data_test_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize
            ])
            batchsize = args.batch_size
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            if args.dset == 'office-home':
                classnames = office_home_classes
            elif args.dset == 'office':
                classnames = office_classes
        
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)
        proto_ini = []


        train_data_dset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        test_data_dset = build_dataset(set_id, data_test_transform, args.data, mode=args.dataset_mode)

       
        print("number of train samples: {}".format(len(train_data_dset)))
        print("number of test samples: {}".format(len(test_data_dset)))


        val_train_loader = torch.utils.data.DataLoader(
                    train_data_dset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                    test_data_dset,
                    batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
        
        results[set_id] = test_time_adapt_eval(val_train_loader,val_loader, model, model_state, optimizer, optim_state, scaler, args,proto_ini)
        del train_data_dset, test_data_dset
        print("finished")
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_time_adapt_eval(val_train_loader,val_loader, model, model_state, optimizer, optim_state, scaler, args,proto_ini):
    memo_bank = {}
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    for cls in range(65):
        memo_bank[cls] = []


    top1_all_init = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    model.eval()
    for i, (images, target) in enumerate(val_loader):
            assert args.gpu is not None
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
            target = target.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output, _, _ = model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_all_init.update(acc1[0], image.size(0))
    print("Init phase")
    print(top1_all_init.avg)
    log_str = '\nInit: {}, Task: {},  Accuracy = {:.2f}%'.format(args.trte, args.name, top1_all_init.avg)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    memo = []
    memo_label = []
    memo_pro = []
    #train
    switch_batch = len(val_train_loader) * args.beta
    print(switch_batch)
    log_str = '\nswitch batch = {}'.format(switch_batch)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    for epoch in range(args.max_epoch):
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top1_all = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        for i, (images, target) in enumerate(val_train_loader):
            assert args.gpu is not None
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
            target = target.cuda(args.gpu, non_blocking=True)
            optimizer.load_state_dict(optim_state)
            if i > switch_batch:
                test_time_tuning_neighbor(model, image,target, optimizer, scaler, args,memo,memo_pro)
            else:
                test_time_tuning(model, image,target, optimizer, scaler, args)


            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output, image_features, _ = model(image)
            image_features = np.array(image_features.detach().cpu()).tolist()
            probs =  output.softmax(dim=-1).detach().cpu().numpy()
            lbl_pseu = torch.argmax(output,1).tolist()
            for idx in range(len(image_features)):
                memo.append(image_features[idx])
                memo_label.append(lbl_pseu[idx])
                memo_pro.append(probs[idx])
                
                
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
        print("training phase")
        print(top1.avg)

        model.eval()
        for i, (images, target) in enumerate(val_loader):
            assert args.gpu is not None


            images = images.cuda(args.gpu, non_blocking=True)
            image = images
            target = target.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output, _, _ = model(image)
                # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_all.update(acc1[0], image.size(0))
            # top5_all.update(acc5[0], image.size(0))
        print("test phase")
        print(top1_all.avg)
        # print(top5_all.avg)
        HA = (2 * top1.avg *top1_all.avg) / ( top1.avg + top1_all.avg)
        log_str = '\nTask: {},  Accuracy on initial = {:.2f}%,  Accuracy on test data test time = {:.2f}%, Accuracy on all test data after = {:.2f}%, Accuracy on HS = {:.2f}%'.format(args.name, top1_all_init.avg,top1.avg,top1_all.avg,HA)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
    
        
        
    return  

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root', default='./TTA_Data/Fine_gra')
    parser.add_argument('--test_sets', type=str, default='Flower102', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # ViT-B/16
    parser.add_argument('-net', '--net', metavar='net', default='ViT')
    parser.add_argument('--step', type=int, default=3, help="times of running")
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--beta', default=0.1, type=float, help='switch the prototype')
    parser.add_argument('--alpha', default=1.0, type=float, help='weight of diverse loss')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--tpt', action='store_true', default= True, help='run zero-shot test time adaptation')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--dset', type=str, default='', choices=['VISDA-C', 'office', 'office-home', 'office-caltech','DomainNet'])
    parser.add_argument('--output_src', type=str, default='./save_model')
    parser.add_argument('--output', type=str, default='./Few_shot_datasets/test')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--val', type=str, default='val_all_50', choices=['all', '0.1'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trte', type=str, default='full', choices=['full', 'val'])
    parser.add_argument('--issave', type=bool, default=True)

    args = parser.parse_args()

    if args.dset == '':
        names = args.test_sets
    
   


    args.output_src = osp.join(args.output_src, args.da, args.val, args.net)
    args.output = osp.join(args.output, args.dset,args.da, args.val, args.net, args.dset, names)
    
    if not osp.exists(args.output_src):
        os.system('mkdir -p ' + args.output_src)
    if not osp.exists(args.output_src):
        os.mkdir(args.output_src)
    if not osp.exists(args.output):
        os.system('mkdir -p ' + args.output)
    if not osp.exists(args.output):
        os.mkdir(args.output)
    
    for i in range(1):
        args.name = names
        args.out_file = open(osp.join(args.output, args.name+'_'+ str(args.beta) +'_step_'+ str(args.step) + '_res.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        main(args)