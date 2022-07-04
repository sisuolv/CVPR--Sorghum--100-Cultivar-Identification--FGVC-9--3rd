
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
# from torchtoolbox.tools import mixup_data, mixup_criterion
# =============================== 初始化 ========================
class Config:
    seed = 2022
    pse_udo = False
    fgvc8 = True
    
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
val_test_transform = A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])


tta_transform0 = A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        A.HorizontalFlip(p=1.0),
        # A.VerticalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])

tta_transform1 =A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        # A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])
tta_transform2 =A.Compose([
        A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
        A.Resize(CFG.image_size, CFG.image_size),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])
tta_transforms = [tta_transform0,tta_transform1,tta_transform2]

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


# =============================== test ========================
def predict_test_raw(net, test_iter):
    '''Inference'''
    net.eval()
    y = []
    net.to(device)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for X in tqdm(test_iter):
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                embeddings = net.extract(X)
                y += softmax(CFG.S * F.linear(F.normalize(embeddings), F.normalize(net.fc.weight))).cpu()
    return np.array(list(Y.numpy() for Y in y))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if CFG.have_index:
        labels_map = {}
        train_df = pd.read_csv(os.path.join(CFG.root_out, 'labels_index.csv'))
        def label_f(m):
            labels_map[int(m.label_index)] = m.cultivar
        train_df.apply(label_f, axis=1)
    else:
        labels_map = data_pre_access(os.path.join(CFG.root_in, 'train_cultivar_mapping.csv'),
                                     output=os.path.join(CFG.root_out, 'labels_index.csv'))
        train_df = pd.read_csv(os.path.join(CFG.root_out, 'labels_index.csv'))
    
    test_df = pd.read_csv('./test.csv')
    test_df['image'] = [CFG.root_in + '/test/' + test_df.iloc[i, 0]  for i in range(test_df.shape[0])]
    
    check_sum = 0
    for key, val in tqdm(labels_map.items()):
        train_df[train_df.label_index == key].cultivar.unique() == val
        check_sum += 1
    check_sum, check_sum == len(labels_map)

    model_lists = [

                    './model/fgvcdata3_resnet50_ibn_a_F_0_Sorghum45_best.params',
                    './model/fgvcdata3_resnet50_ibn_b_F_0_Sorghum45_best.params',
                    './model/fgvcdata_resnet50_ibn_a_F_0_Sorghum45_best.params',
                    './model/fgvcdata_resnet50_ibn_b_F_0_Sorghum45_best.params',
                    './model/fgvcdata_resnet50_ibn_a_F_1_Sorghum45_best.params',

                   ]

    for tt in range(len(model_lists)):
        # # Load model
        if 'resnet50_ibn_a' in model_lists[tt]:
            CFG.model_name = 'resnet50_ibn_a'
        elif 'resnet50_ibn_b' in model_lists[tt]:
            CFG.model_name = 'resnet50_ibn_b'
        else:
            print('no model')

        model = SorghumModel(CFG.model_name, CFG.embedding_size, CFG.num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_lists[tt]))
        if tt == 0:
            # # no TTA
            sorghum_test_dataset = Sorghum_Test_Dataset(df=test_df, transform=val_test_transform)
            sorghum_test_loader = DataLoader(sorghum_test_dataset, batch_size=CFG.batch_size, shuffle=False,
                                             num_workers=CFG.num_workers)
            result_raw_original = predict_test_raw(model, sorghum_test_loader)
            np.save(os.path.join(CFG.root_out, 'test_result_raw_Original.npy'), result_raw_original)

            # ## Save TTA result
            result_raw_original = np.load(os.path.join(CFG.root_out, 'test_result_raw_Original.npy'))
            result_raw_ttas = {'origin': result_raw_original, 'avg': result_raw_original}

            for i in range(len(tta_transforms)):
                torch.cuda.empty_cache()
                sorghum_test_dataset = Sorghum_Test_Dataset(df=test_df, transform=tta_transforms[i])
                sorghum_test_loader = DataLoader(sorghum_test_dataset, batch_size=CFG.batch_size, shuffle=False,
                                                 num_workers=CFG.num_workers)
                result_raw_tta = predict_test_raw(model, sorghum_test_loader)
                np.save(os.path.join(CFG.root_out, 'test_result_raw_' + 'tta_' + str(i) + '_.npy'), result_raw_tta)
                result_raw_ttas['tta_' + str(i)] = result_raw_tta
                result_raw_ttas['avg'] += result_raw_tta

            result_raw_ttas['avg'] /= len(result_raw_ttas.keys()) - 1
        else:
            # # no TTA
            sorghum_test_dataset = Sorghum_Test_Dataset(df=test_df, transform=val_test_transform)
            sorghum_test_loader = DataLoader(sorghum_test_dataset, batch_size=CFG.batch_size, shuffle=False,
                                             num_workers=CFG.num_workers)
            result_raw_original2 = predict_test_raw(model, sorghum_test_loader)
            np.save(os.path.join(CFG.root_out, 'test_result_raw_Original2.npy'), result_raw_original2)

            # ## Save TTA result
            result_raw_original2 = np.load(os.path.join(CFG.root_out, 'test_result_raw_Original2.npy'))
            result_raw_ttas2 = {'origin': result_raw_original2, 'avg': result_raw_original2}


            for i in range(len(tta_transforms)):
                torch.cuda.empty_cache()
                sorghum_test_dataset = Sorghum_Test_Dataset(df=test_df, transform=tta_transforms[i])
                sorghum_test_loader = DataLoader(sorghum_test_dataset, batch_size=CFG.batch_size, shuffle=False,
                                                 num_workers=CFG.num_workers)
                result_raw_tta = predict_test_raw(model, sorghum_test_loader)
                np.save(os.path.join(CFG.root_out, 'test_result_raw_' + 'tta_' + str(i) + '_.npy'), result_raw_tta)
                result_raw_ttas2['tta_' + str(i)] = result_raw_tta
                result_raw_ttas2['avg'] += result_raw_tta

            result_raw_ttas2['avg'] /= len(result_raw_ttas2.keys()) - 1

            result_raw_ttas['avg'] += result_raw_ttas2['avg']

    result_raw_ttas['avg'] = result_raw_ttas['avg']/len(model_lists)

    result_ttas_sorted_val = {}
    result_ttas_sorted_idx = {}
    result_raw_ttas.keys()

    for key, val in result_raw_ttas.items():
        torch.cuda.empty_cache()
        result_tta = torch.tensor(val, dtype=torch.float32, device='cuda')
        result_sorted_val, result_sorted_idx = result_tta.sort(dim=1, descending=True)
        result_ttas_sorted_val[key] = result_sorted_val.cpu().numpy()
        result_ttas_sorted_idx[key] = result_sorted_idx.cpu().numpy()
        del result_tta, result_sorted_val, result_sorted_idx

    # ## Predict and make submission
    result_sorted_val = result_ttas_sorted_val['avg']
    result_sorted_idx = result_ttas_sorted_idx['avg']





    result = pd.read_csv('./fgvcdata/sample_submission.csv')
    result['cultivar'] = [labels_map.get(result_sorted_idx[i, 0]) for i in range(result_sorted_idx.shape[0])]
    result = result.set_index('filename')
    result.to_csv(os.path.join(CFG.root_out, 'submission.csv'))