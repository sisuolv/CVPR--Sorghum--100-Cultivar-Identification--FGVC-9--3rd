
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image as Img
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
import timm
import torchvision
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math

from resnet_ibn_a import *
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
from torchtoolbox.tools import mixup_data, mixup_criterion
# from transformers import get_linear_schedule_with_warmup

# =============================== 初始化 ========================
class Config:
    seed = 2022
    pse_udo = False
    fgvc8 = False
    
    attack_awp = True
    attack_start_epoch = 0.0

    use_amp = True
    num_workers = 4
    fold = 5
    model_name ='resnet50_ibn_a'# 'resnet50_ibn_a'   #'tf_efficientnet_b3_ns'
    optim = 'adamW'
    lr = 1e-4
    weight_decay = 1e-3
    eta_min = 1e-5

    batch_size = 12
    image_size = 1024
    epoch = 45


    root_in = './fgvcdata'  # Folder with input (image, lable)
    root_out = './'  # Folder with output (csv, pth)
    have_index = True  # If the breed label have been map to a index

    '''ArcFace parameter'''
    num_classes = 100
    embedding_size = 1024
    S, M = 30.0, 0.3  # S:consine scale in arcloss. M:arg penalty
    EASY_MERGING, LS_EPS = False, 0.0
    
    '''mixup parameter'''
    mix_up = True
    mixup_prob = 0.25    
    alpha = 1
    
    '''cutmix parameter'''
    cut_mix = True
    cutmix_prob = 0.25
    beta = 1

CFG = Config()

def seed_it(seed):
    #     random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(CFG.seed)

# =============================== help function========================
class Accumulator():
    '''A counter util, which count the float value of the input'''
    def __init__(self, nums):
        self.metric = list(torch.zeros((nums,)).numpy())
    def __getitem__(self, index):
        return self.metric[index]
    def add(self, *args):
        for i, item in enumerate(args):
            self.metric[i] += float(item)


def accuracy(y_hat, y):
    '''used to count the right type'''
    y_hat = y_hat.exp().argmax(dim=1)
    y_hat.reshape((-1))
    y.reshape((-1))
    return accuracy_score(y.cpu().numpy(), y_hat.cpu().numpy(), normalize=False)


def evaluate_accuracy(net, data_iter):
    '''Evalue the valid dataset'''
    net.eval()
    softmax = nn.Softmax(dim=1)
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y.to(device)
            #             y_hat = net(X, y)
            with torch.cuda.amp.autocast(enabled=True):
                embeddings = net.extract(X)
                y_hat = softmax(CFG.S * F.linear(F.normalize(embeddings), F.normalize(net.fc.weight)))
                # y_hat = net(X, y)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def data_pre_access(file, output):
    '''transfer train label into index'''
    labels = pd.read_csv(file, index_col='image')
    labels_map = dict()
    labels['label_index'] = torch.zeros((labels.shape[0])).type(torch.int32).numpy()
    for i, label in enumerate(labels.cultivar.unique()):
        labels_map[i] = label
        labels.loc[labels.cultivar == label, 'label_index'] = i
    labels.to_csv(output)
    return labels_map

# =============================== Dataset and  transform========================

train_transform =  A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])


val_test_transform = A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])



class Sorghum_Train_Dataset(Dataset):
    '''Train Dataset'''

    def __init__(self, img_path_csv='', df=None, transform=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(img_path_csv)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img = Img.open(self.df.iloc[index, 0])
        img = np.array(img)
        
        if 'fgvc8' in self.df.iloc[index, 0]:
            tmp1 = np.concatenate([img, img, img[0:64, :, :]], axis=0)
            img = np.concatenate([tmp1, tmp1, tmp1[:, 0:64, :]], axis=1)
            # print(img.shape)
            
        label_index = self.df.iloc[index, 1]
        if self.transform is not None:
            img = self.transform(image=img)
            img = img['image']
        return img, label_index


class Sorghum_Test_Dataset(Sorghum_Train_Dataset):
    '''Test Dataset'''

    def __getitem__(self, index):
        img = Img.open(self.df.iloc[index, 0])
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)
            img = img['image']
        return img



# =============================== model ========================
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SorghumModel(nn.Module):
    def __init__(self, model_name, embedding_size, num_classes, pretrained=True):
        super(SorghumModel, self).__init__()

        if 'efficientnet' in model_name:
            self.model = timm.create_model(model_name, in_chans=3, pretrained=pretrained, num_classes=num_classes)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.pooling = self.model.global_pool
            self.model.global_pool = nn.Identity()
        elif  model_name == 'resnet50': 
            self.model = timm.create_model(model_name, in_chans=3, pretrained=pretrained,  num_classes=0, global_pool='')
            in_features = 2048
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif model_name == 'resnet50_ibn_a':
            self.model = resnet50_ibn_a(pretrained=pretrained)
            in_features =2048
            self.pooling = nn.AdaptiveAvgPool2d(1)
            # self.pooling = GeM()
        elif model_name == 'resnet50_ibn_b':
            self.model = resnet50_ibn_b(pretrained=pretrained)
            in_features = 2048
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif model_name == 'resnet101_ibn_a':
            self.model = resnet101_ibn_a(pretrained=pretrained)
            in_features =2048
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif model_name == 'resnet101_ibn_b':
            self.model = resnet101_ibn_b(pretrained=pretrained)
            in_features = 2048
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.enhance = se_block(channel=in_features, ratio=8)
        self.multiple_dropout = [nn.Dropout(0.25) for i in range(5)]
        self.embedding = nn.Linear(in_features, embedding_size)
        #bnnneck
        self.bottleneck = nn.BatchNorm1d(embedding_size)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.pr = torch.nn.PReLU()
        
        self.fc = ArcMarginProduct(embedding_size, num_classes, CFG.S, CFG.M, CFG.EASY_MERGING, CFG.LS_EPS)

    def forward(self, images, labels):
        features = self.model(images)
        features = self.enhance(features)
        pooled_features = self.pooling(features).flatten(1)
        pooled_features_dropout = torch.zeros((pooled_features.shape)).to(device)
        for i in range(5):
            pooled_features_dropout += self.multiple_dropout[i](pooled_features)
        pooled_features_dropout /= 5
        embedding = self.embedding(pooled_features_dropout)
        embedding = self.bottleneck(embedding)
        embedding = self.pr(embedding)
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        features = self.model(images)
        features = self.enhance(features)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        embedding = self.bottleneck(embedding)
        embedding = self.pr(embedding)
        return embedding
# ===============================ArcFace ========================
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            s: float,
            m: float,
            easy_margin: bool,
            ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long().to(device), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output   
# ===============================train model========================
def train_model(net, loss):
    num_batches = len(train_loader)
    best_accuracy = 0
    
    if CFG.optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=CFG.lr,
                                    weight_decay=CFG.weight_decay)
    elif CFG.optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=CFG.lr,
                                     weight_decay=CFG.weight_decay)
    elif CFG.optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=CFG.lr,
                                      weight_decay=CFG.weight_decay)
    elif CFG.optim == 'ranger':
        optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=CFG.lr,
                           weight_decay=CFG.weight_decay)

        # scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_amp)  # mixed_precison
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=CFG.eta_min,
                                                                         last_epoch=-1)
    
#     num_total_steps = len(train_loader) * CFG.epoch
#     CFG.warmup_steps = num_total_steps*0.1
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
#                                                 num_training_steps=num_total_steps)
    if CFG.attack_awp:
        # attack_func = AWP(net)
        print('Enable AWP_fast')
        attack_func = AWP_fast(net, optimizer, adv_lr=0.001, adv_eps=0.001)
        
    for epoch in range(CFG.epoch):
        net.train()
        metric = Accumulator(3)
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            targets = targets.to(device)
            if CFG.attack_awp:                   
                if epoch >= CFG.attack_start_epoch:
                    attack_func.perturb()
                    
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                # mixup and cutmix       
                rand = np.random.rand()
                if CFG.mix_up and (rand < CFG.mixup_prob):    
                    images, labels_a, labels_b, lam = mixup_data(images, targets, CFG.alpha)
                    y_hat = net(images, targets)
                    l = mixup_criterion(loss, y_hat, labels_a, labels_b, lam)
                elif CFG.cut_mix and (CFG.mixup_prob< rand < (CFG.mixup_prob + CFG.cutmix_prob)):
                    lam = np.random.beta(CFG.beta, CFG.beta)
                    rand_index = torch.randperm(images.size()[0])
                    target_a = targets
                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    # compute output
                    y_hat = net(images, targets)
                    l = loss(y_hat, target_a) * lam + loss(y_hat, target_b) * (1. - lam)
                else:
                    y_hat = net(images, targets)
                    l = loss(y_hat, targets)

            # scaler.scale(l).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            l.backward()
            if CFG.attack_awp: 
                attack_func.restore()  
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            
            with torch.no_grad():
                metric.add(l * images.shape[0], accuracy(y_hat, targets), images.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 1) == 0 or i == num_batches - 1:
                print(epoch + (i + 1) / num_batches, 'train_l train_acc\t', (train_l, train_acc, None))
                writer.add_scalars('Loss/Accuracy/train/Fold-' + str(k_fold),
                                   {'train_accuracy': np.array(train_acc), 'train_loss': np.array(train_l)},
                                   10 * np.array(epoch + (i + 1) / num_batches))
                Value_train_l.append(train_l)
                Value_train_acc.append(train_acc)
                Value_test_acc.append(None)
                Time.append(epoch + (i + 1) / num_batches)

        scheduler.step()

        test_acc = evaluate_accuracy(net, val_loader)

        print('lr = ', optimizer.param_groups[0]['lr'], epoch + 1, 'test_acc\t', (None, None, test_acc))
        writer.add_scalars('Loss/Accuracy/test/Fold-' + str(k_fold), {'val_accuracy': np.array(test_acc)},
                           10 * np.array(epoch + 1))
        Value_train_l.append(None)
        Value_train_acc.append(None)
        Value_test_acc.append(test_acc)
        Time.append(epoch + 1)
        if test_acc >= best_accuracy:
            best_accuracy = test_acc
            torch.save(net.state_dict(),
                       os.path.join(CFG.root_out, sub_fold, 'Sorghum' + str(epoch + 1) + '_best.params'))
        
        torch.save(net.state_dict(),
                       os.path.join(CFG.root_out, sub_fold, 'Sorghum' + str(epoch + 1) + '_best.params'))

        record_data = pd.DataFrame(zip(Value_train_l, Value_train_acc, Value_test_acc, Time))
        record_data.to_csv(os.path.join(CFG.root_out, sub_fold, 'Record_Sorghum.csv'))
        
    torch.save(net.state_dict(), os.path.join(CFG.root_out, sub_fold, 'Sorghum' + str(epoch + 1) + '_last.params'))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' f'test acc {test_acc:.3f}')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if CFG.have_index:
        labels_map = {}
        train_df = pd.read_csv(os.path.join(CFG.root_out, 'labels_index.csv'))
        def label_f(m):
            labels_map[int(m.label_index)] = m.cultivar
        train_df.apply(label_f, axis=1)
        def get_key(dct, value):
            return [k for (k,v) in dct.items() if v == value][0]
        train_df['image'] = [CFG.root_in + '/train/' + train_df.iloc[i, 0]  for i in range(train_df.shape[0])]
        train_df['cultivar'] = [get_key(labels_map, train_df.iloc[i, 1]) for i in range(train_df.shape[0])]
        train_df = train_df.iloc[:, 0:2]
        train_df.rename(columns={'cultivar':'label_index'}, inplace=True)
    else:
        labels_map = data_pre_access(os.path.join(CFG.root_in, 'train_cultivar_mapping.csv'),
                                     output=os.path.join(CFG.root_out, 'labels_index.csv'))
        train_df = pd.read_csv(os.path.join(CFG.root_out, 'labels_index.csv'))
    
    
    if CFG.pse_udo:
        def get_key(dct, value):
            return [k for (k,v) in dct.items() if v == value][0]
        test_df = pd.read_csv(os.path.join(CFG.root_in, 'test_pseudo.csv'))
        test_df['cultivar'] = [get_key(labels_map, test_df.iloc[i, 1]) for i in range(test_df.shape[0])]
        test_df.rename(columns={'filename':'image'}, inplace=True)
        test_df['image'] = [CFG.root_in + '/test/' + test_df.iloc[i, 0].replace('png','jpeg')  for i in range(test_df.shape[0])]
        
        test_df.rename(columns={'cultivar':'label_index'}, inplace=True)
        train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    if CFG.fgvc8:
        def get_key(dct, value):
            return [k for (k,v) in dct.items() if v == value][0]
        fgvc8_df = pd.read_csv(os.path.join(CFG.root_in, 'fgvc8_data.csv'))
        fgvc8_df['cultivar'] = [get_key(labels_map, fgvc8_df.iloc[i, 1]) for i in range(fgvc8_df.shape[0])]
        fgvc8_df.rename(columns={'path':'image'}, inplace=True)
        fgvc8_df['image'] = [CFG.root_in + '/' + fgvc8_df.iloc[i, 0][2:] for i in range(fgvc8_df.shape[0])]
        
        fgvc8_df.rename(columns={'cultivar':'label_index'}, inplace=True)
        train_df = pd.concat([train_df, fgvc8_df], axis=0).reset_index(drop=True)
        
    print(train_df)
    
    sfolder = StratifiedKFold(n_splits=CFG.fold, random_state=CFG.seed, shuffle=True)
    train_folds = []
    val_folds = []
    for train_idx, val_idx in sfolder.split(train_df.image, train_df.label_index):
        train_folds.append(train_idx)
        val_folds.append(val_idx)

    os.makedirs('/root/tf-logs/' + CFG.model_name, exist_ok=True)
    writer = SummaryWriter(log_dir='/root/tf-logs/' + CFG.model_name)

    for k_fold in range(CFG.fold):
        if k_fold >= 1:
            break
        print('\n ********** Fold %d **********\n' % k_fold)
        Value_train_l = list()
        Value_train_acc = list()
        Value_test_acc = list()
        Time = list()
        sub_fold = CFG.model_name + '_F_' + str(k_fold)
        os.makedirs(os.path.join(CFG.root_out, sub_fold), exist_ok=True)

        # train_dataset = Sorghum_Train_Dataset(df=train_df.iloc[train_folds[k_fold]], transform=train_transform)
        train_dataset = Sorghum_Train_Dataset(df=train_df, transform=train_transform)
        
        if CFG.pse_udo:
            val_dataset = Sorghum_Train_Dataset(df=test_df, transform=val_test_transform)
        else:
            val_dataset = Sorghum_Train_Dataset(df=train_df.iloc[val_folds[k_fold]], transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

        net = SorghumModel(CFG.model_name, CFG.embedding_size, CFG.num_classes, pretrained=True).to(device)
        loss = nn.CrossEntropyLoss()
        train_model(net, loss)
    writer.close()
