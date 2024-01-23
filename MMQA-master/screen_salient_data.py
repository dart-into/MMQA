from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings("ignore")
import random
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True

import re
import torch.nn.functional as F
#from utils import load_state_dict_from_url
from collections import OrderedDict

import numpy as np
from scipy.ndimage import sobel
from skimage import img_as_float
from scipy.stats import genpareto
from scipy.special import gammaln
from scipy.ndimage import filters


def Salient(patch_size, stride):
    # find the min gradient patch
    #def __init__(self, patch_size, stride):
        
        source_folder = "/home/user/data/Kadid/kadid10k/images"
        max_folder = "/home/user/data/Kadid/kadid10k/salient_images"
        min_folder = "/home/user/data/Kadid/kadid10k/non_salient_images"

        image_files = os.listdir(source_folder)
        #print(image_files)
        os.makedirs(max_folder, exist_ok=True)
        os.makedirs(min_folder, exist_ok=True)
        
        work = 0
        for image_file in image_files[1:]:
            #print(image_file)
            source_path = os.path.join(source_folder, image_file)
            max_path = os.path.join(max_folder, image_file)
            min_path = os.path.join(min_folder, image_file)
            #print(source_path)

            
            patch = []
            im = Image.open(source_path).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            image_height, image_width, _ = image.shape
            features = []
            for i in range(0, image_height - patch_size + 1, stride):
                for j in range(0, image_width - patch_size + 1, stride):
                    block = image[i:i + patch_size, j:j + patch_size]
                    patch.append(block)

                    gradient_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                    gradient_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                    block = np.sqrt(gradient_x**2 + gradient_y**2)

                    mu = np.mean(block)
                    sigma = np.std(block)
                    filtered_block = filters.gaussian_filter(block, sigma)
                    shape, _, scale = genpareto.fit(filtered_block.ravel(), floc=0)
                    feature = [mu, sigma, shape, scale, gammaln(1 / shape)]
                    features.append(feature)
 
            features = np.array(features)
            model_mean = np.zeros(features.shape[1])
            model_cov_inv = np.eye(features.shape[1])
            count = 0
            min_num = 0
            max_num = 0
            niqe = []
            idex_max = 0
            idex_min = 0
            for i,feature in enumerate(features):
                score = (feature - model_mean) @ model_cov_inv @ (feature - model_mean).T

                if i ==0:
                    min_num=score

                if score < min_num:
                    min_num = score
                    idex_min = count
            
                if score > max_num:
                    max_num = score
                    idex_max = count

                count = count+1

        #print('min_niqe', niqe, idex)
        
            minpatch = patch[idex_min]
            maxpatch = patch[idex_max]
        #cv2.imwrite('/home/b19190428/GXL/min_niqe_patch.jpg', minpatch)

            cv2.imwrite(min_path, minpatch)
            cv2.imwrite(max_path, maxpatch)

            work = work + 1
            print('num %d is done' %work)


Salient(224, 112)
print('Good job, it is well done!!!')
        