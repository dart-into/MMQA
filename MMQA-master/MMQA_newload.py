from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2
from tqdm import tqdm

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
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True

import one_load
import two_load
import three_load
import mergenet

import re
import torch.nn.functional as F
#from utils import load_state_dict_from_url
from collections import OrderedDict



class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h == new_h and w == new_w:
            return {'image': image, 'rating': rating}

        elif h == new_h and w != new_w:
            left = np.random.randint(0, w - new_w)
            image = image[0: new_h,
                left: left + new_w]
            return {'image': image, 'rating': rating}

        elif h != new_h and w == new_w:
            top = np.random.randint(0, h - new_h)
            image = image[top: top + new_h,
                0:  new_w]
            return {'image': image, 'rating': rating}

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}



class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)              #add norm
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        #print('L9 size',out.size())
        out = self.bn3(out)
        out = self.sig(out)
        #print('L10',out)
        #print('L10 size',out.size())
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out


class Net(nn.Module):
    def __init__(self , net1, net2, net3, mergenet, linear):
        super(Net, self).__init__()
        self.Net1 = net1
        self.Net2 = net2
        self.Net3 = net3
        self.Merge = mergenet
        self.Linear = linear

    def forward(self, x1, x2, x3):
        x1 = self.Net1(x1)
        x2 = self.Net2(x2)
        x3 = self.Net3(x3)
        features = torch.cat((x1, x2, x3), dim=1)
        x = self.Merge(features)
        x = self.Linear(x)
        return x


def computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for data1, data2, data3 in zip(dataloader_valid1, dataloader_valid2, dataloader_valid3):
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating'].view(batch_size1, -1)
            inputs2 = data2['image']
            batch_size2 = inputs2.size()[0]
            labels2 = data2['rating'].view(batch_size2, -1)
            inputs3 = data3['image']
            batch_size3 = inputs3.size()[0]
            labels3 = data3['rating'].view(batch_size3, -1)


            if use_gpu:
                try:
                    inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                    inputs2 = Variable(inputs2.float().cuda())
                    inputs3 = Variable(inputs3.float().cuda())
                except:
                    print(inputs1, labels1, inputs2, labels2, inputs3, labels3)
            else:
                inputs1, labels1 = Variable(inputs1), Variable(labels1)
                inputs2 = Variable(inputs2)
                inputs3 = Variable(inputs3)

            outputs_a = model(inputs1, inputs2, inputs3)
            ratings.append(labels1.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)
    return sp

def train_model():
    epochs = 5
    task_num = 5
    noise_num1 = 24
    noise_num2 = 25

    net_1 = one_load.densenetnew(pretrained=False)
    net_2 = two_load.densenetnew(pretrained=False)
    net_3 = three_load.densenetnew(pretrained=False)
    m_net = mergenet.merge_net(pretrained=False)
    l_net = BaselineModel1(1, 0.5, 1000)

    densenet_model = models.densenet121(pretrained = True)
    state_dict = densenet_model.features.state_dict()

    for name in list(state_dict.keys()):
        if name.startswith('denseblock4.'):
            del state_dict[name]
        '''
        if name.startswith('transition3.'):
            del state_dict[name]
        '''
        if name.startswith('norm5.'):
            del state_dict[name]
    #print(list(state_dict.keys()))
    net_1.features.load_state_dict(state_dict)
    net_2.features.load_state_dict(state_dict)
    net_3.features.load_state_dict(state_dict)

    model = Net(net1 = net_1, net2 = net_2, net3 = net_3, mergenet = m_net, linear = l_net)
    #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_densenet_new.pt'))


    criterion = nn.MSELoss()
    ignored_params = list(map(id, model.Linear.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.Linear.parameters(), 'lr': 1e-2}
    ], lr=1e-4)
    model.cuda()
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)

    spearman = 0

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)

        list_noise = list(range(noise_num1))
        np.random.shuffle(list_noise)
        print('============= TID2013 train phase epoch %2d ==============' % epoch)
        count = 0
        for index in list_noise:

            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            dataloader_train1, dataloader_valid1 = load_data('train1', 'tid2013', index)         
            dataloader_train2, dataloader_valid2 = load_data('train2', 'tid2013', index)
            dataloader_train3, dataloader_valid3 = load_data('train3', 'tid2013', index)

            if dataloader_train1 == 0:
                continue

            dataiter1 = iter(enumerate(dataloader_valid1))
            dataiter2 = iter(enumerate(dataloader_valid2))
            dataiter3 = iter(enumerate(dataloader_valid3))
            model.train()  # Set model to training mode
            # Iterate over data.

            total_iterations = len(dataloader_train1)
            for data1, data2, data3 in  tqdm(zip(dataloader_train1, dataloader_train2, dataloader_train3), total=total_iterations, desc='Processing'):
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                #print('input1', inputs1)
                # labels = labels / 10.0
                inputs2 = data2['image']
                batch_size2 = inputs2.size()[0]
                labels2 = data2['rating'].view(batch_size2, -1)
                #print('input2', inputs2)
                inputs3 = data3['image']
                batch_size3 = inputs3.size()[0]
                labels3 = data3['rating'].view(batch_size3, -1)
                #print('input3', inputs3)

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        inputs2 = Variable(inputs2.float().cuda())
                        inputs3 = Variable(inputs3.float().cuda())
                    except:
                        print(inputs1, labels1, inputs2, labels2, inputs3, labels3)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)
                    inputs2 = Variable(inputs2)
                    inputs3 = Variable(inputs3)

                optimizer.zero_grad()
                outputs = model(inputs1, inputs2, inputs3)
                #print('outputs', outputs)
                loss = criterion(outputs, labels1)
                #print('labels1', labels1)
                loss.backward()
                optimizer.step()

                idx1, data_val1 = next(dataiter1)
                idx2, data_val2 = next(dataiter2)
                idx3, data_val3 = next(dataiter3)
                if idx1 >= len(dataloader_valid1)-1:
                    dataiter1 = iter(enumerate(dataloader_valid1))
                    dataiter2 = iter(enumerate(dataloader_valid2))
                    dataiter3 = iter(enumerate(dataloader_valid3))
                inputs_val1 = data_val1['image']
                batch_size1 = inputs_val1.size()[0]
                labels_val1 = data_val1['rating'].view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                inputs_val2 = data_val2['image']
                batch_size2 = inputs_val2.size()[0]
                labels_val2 = data_val2['rating'].view(batch_size2, -1)
                inputs_val3 = data_val3['image']
                batch_size3 = inputs_val3.size()[0]
                labels_val3 = data_val3['rating'].view(batch_size3, -1)
                if use_gpu:
                    try:
                        inputs_val1, labels_val1 = Variable(inputs_val1.float().cuda()), Variable(labels_val1.float().cuda())
                        inputs_val2 = Variable(inputs_val2.float().cuda())
                        inputs_val3 = Variable(inputs_val3.float().cuda())
                    except:
                        print(inputs_val1, labels_val1, inputs_val2, inputs_val3)
                else:
                    inputs_val1, labels_val1 = Variable(inputs_val1), Variable(labels_val1)
                    inputs_val2 = Variable(inputs_val2)
                    inputs_val3 = Variable(inputs_val3)

                optimizer.zero_grad()
                outputs_val = model(inputs_val1, inputs_val2, inputs_val3)
                loss_val = criterion(outputs_val, labels_val1)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param1 = dict(meta_model.named_parameters())
                name_to_param2 = dict(temp_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param2[name].data
                    name_to_param1[name].data.add_(diff / task_num)

                count += 1
        epoch_loss = running_loss / count
        print('current loss = ',epoch_loss)

        
        running_loss = 0.0
        list_noise = list(range(noise_num2))
        np.random.shuffle(list_noise)
        print('=============== Kadid train phase epoch %2d =================' % epoch)
        count = 0
        for index in list_noise:
            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)

            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            #dataloader_train, dataloader_valid = load_data('train', 'kadid10k', index)
            dataloader_train1, dataloader_valid1 = load_data('train1', 'kadid10k', index)
            dataloader_train2, dataloader_valid2 = load_data('train2', 'kadid10k', index)
            dataloader_train3, dataloader_valid3 = load_data('train3', 'kadid10k', index)

            if dataloader_train1 == 0:
                continue

            dataiter1 = iter(enumerate(dataloader_valid1))
            dataiter2 = iter(enumerate(dataloader_valid2))
            dataiter3 = iter(enumerate(dataloader_valid3))
            model.train()  # Set model to training mode
            # Iterate over data.

            total_iterations = len(dataloader_train1)
            for data1, data2, data3 in tqdm(zip(dataloader_train1, dataloader_train2, dataloader_train3), total=total_iterations, desc='Processing'):
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                # labels = labels / 10.0
                inputs2 = data2['image']
                batch_size2 = inputs2.size()[0]
                labels2 = data2['rating'].view(batch_size2, -1)
                inputs3 = data3['image']
                batch_size3 = inputs3.size()[0]
                labels3 = data3['rating'].view(batch_size3, -1)
                labels1 = (labels1 - 0.5) / 5.0

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        inputs2 = Variable(inputs2.float().cuda())
                        inputs3 = Variable(inputs3.float().cuda())
                    except:
                        print(inputs1, labels1, inputs2, labels2, inputs3, labels3)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)
                    inputs2 = Variable(inputs2)
                    inputs3 = Variable(inputs3)

                optimizer.zero_grad()
                outputs = model(inputs1, inputs2, inputs3)
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()

                idx1, data_val1 = next(dataiter1)
                idx2, data_val2 = next(dataiter2)
                idx3, data_val3 = next(dataiter3)
                if idx1 >= len(dataloader_valid1)-1:
                    dataiter1 = iter(enumerate(dataloader_valid1))
                    dataiter2 = iter(enumerate(dataloader_valid2))
                    dataiter3 = iter(enumerate(dataloader_valid3))
                inputs_val1 = data_val1['image']
                batch_size1 = inputs_val1.size()[0]
                labels_val1 = data_val1['rating'].view(batch_size1, -1)
                # labels_val = labels_val / 10.0
                inputs_val2 = data_val2['image']
                batch_size2 = inputs_val2.size()[0]
                labels_val2 = data_val2['rating'].view(batch_size2, -1)
                inputs_val3 = data_val3['image']
                batch_size3 = inputs_val3.size()[0]
                labels_val3 = data_val3['rating'].view(batch_size3, -1)
                labels_val1 = (labels_val1 - 0.5) / 5.0

                if use_gpu:
                    try:
                        inputs_val1, labels_val1 = Variable(inputs_val1.float().cuda()), Variable(labels_val1.float().cuda())
                        inputs_val2 = Variable(inputs_val2.float().cuda())
                        inputs_val3 = Variable(inputs_val3.float().cuda())
                    except:
                        print(inputs_val1, labels_val1, inputs_val2, inputs_val3)
                else:
                    inputs_val1, labels_val1 = Variable(inputs_val1), Variable(labels_val1)
                    inputs_val2 = Variable(inputs_val2)
                    inputs_val3 = Variable(inputs_val3)

                optimizer.zero_grad()
                outputs_val = model(inputs_val1, inputs_val2, inputs_val3)
                loss_val = criterion(outputs_val, labels_val1)
                loss_val.backward()
                optimizer.step()

                try:
                    running_loss += loss_val.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                name_to_param = dict(meta_model.named_parameters())
                for name, param in model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff / task_num)

                count += 1
        # print('trying epoch loss')
        epoch_loss = running_loss / count
        print('current loss = ',epoch_loss)
        

        print('############# test phase epoch %2d ###############' % epoch)
        dataloader_train1, dataloader_valid1 = load_data('test1', 0)
        dataloader_train2, dataloader_valid2 = load_data('test2', 0)
        dataloader_train3, dataloader_valid3 = load_data('test3', 0)
        model.eval()
        model.cuda()
        sp = computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model)[0]
        if sp > spearman:
            spearman = sp
            best_model = copy.deepcopy(model)
            #best_model = copy.deepcopy(meta_model)
            # torch.save(best_model.cuda(),
            #        'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
        
        '''
        for k,v in model.state_dict().items():
            with open('record.txt', 'a')as file:
                file.write(f"the {epoch} record:{k}{v}\n")
                #file.save()
                file.close()
        '''
        #for param_tensor in model.state_dict():
        #    print(param_tensor, model.state_dict()[param_tensor])
        print('new srocc {:4f}, best srocc {:4f}'.format(sp, spearman))
    torch.save(model.cuda().state_dict(), 
               'model_IQA/TID2013_KADID10K_IQA_Meta_densenet_newload.pt')
    #torch.save(model.cuda(),
    #       'model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
    

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=2):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.9**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train', dataset = 'tid2013', worker_idx = 0):

    if dataset == 'tid2013':
        data_dir = os.path.join('/home/user/MMQA-master/tid2013')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path1 = '/home/user/data/tid2013/distorted_images/'
        image_path2 = '/home/user/data/tid2013/salient_images/'
        image_path3 = '/home/user/data/tid2013/non_salient_images/'
    else:
        data_dir = os.path.join('/home/user/MMQA-master/kadid10k')
        worker_orignal = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_per_noise.csv'), sep=',')
        image_path1 = '/home/user/data/Kadid/kadid10k/images/'
        image_path2 = '/home/user/data/Kadid/kadid10k/salient_images/'
        image_path3 = '/home/user/data/Kadid/kadid10k/non_salient_images/'
    workers_fold = "noise/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    #worker = worker_orignal['noise'].unique()[worker_idx]
    #print("----worker number: %2d---- %s" %(worker_idx, worker))
    worker = worker_orignal['noise'].unique()[worker_idx]
    
    percent = 0.8
    images = worker_orignal[worker_orignal['noise'].isin([worker])][['image', 'dmos']]

    train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
    train_path = workers_fold + "train_scores_" + str(worker) + ".csv"
    test_path = workers_fold + "test_scores_" + str(worker) + ".csv"
    train_dataframe.to_csv(train_path, sep=',', index=False)
    valid_dataframe.to_csv(test_path, sep=',', index=False)

    if mod == 'train1':

        #worker = worker_orignal['noise'].unique()[worker_idx]
        print("----worker number: %2d---- %s" %(worker_idx, worker))


        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path1,
                                                        transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                      RandomHorizontalFlip(0.5),
                                                                                      RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path1,
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=16, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=10, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    elif mod == 'train2':

        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path2,
                                                        transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path2,
                                                        transform=transforms.Compose([#RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=16, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=10, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    elif mod == 'train3':

        output_size = (224, 224)
        transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                        root_dir=image_path3,
                                                        transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                      #RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                        root_dir=image_path3,
                                                        transform=transforms.Compose([#RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=16, drop_last=True,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
        dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=10, drop_last=True,
                                      shuffle=False, num_workers=0, collate_fn=my_collate)

    
    elif mod == 'test1':
        #worker = worker_orignal['noise'].unique()[worker_idx]
        output_size = (224, 224)
        print("----worker number: %2d---- %s" %(worker_idx, worker))
        cross_data_path = '/home/user/MetaIQA-master/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/b19190428/data/LIVEwild/ChallengeDB_release/Images',
                                                        transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= 10,
                                        shuffle=False, num_workers=0)

    elif mod == 'test2':
        output_size = (224, 224)
        cross_data_path = '/home/user/MetaIQA-master/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/b19190428/data/LIVEwild/ChallengeDB_release/salient_images',
                                                        transform=transforms.Compose([#NIQEMax(patch_size=224, stride=112),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= 10,
                                        shuffle=False, num_workers=0)


    else:
        output_size = (224, 224)
        cross_data_path = '/home/user/MetaIQA-master/LIVE_WILD/image_labeled_by_score.csv'
        transformed_dataset_valid_1 = ImageRatingsDataset(csv_file=cross_data_path,
                                                        root_dir='/home/b19190428/data/LIVEwild/ChallengeDB_release/non_salient_images',
                                                        transform=transforms.Compose([#NIQEMin(patch_size=224, stride=112),
                                                                                      #RandomCrop(output_size=output_size),
                                                                                      Normalize(),
                                                                                      ToTensor(),
                                                                                      ]))
        dataloader_train = 0
        dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size= 10,
                                        shuffle=False, num_workers=0)


    return dataloader_train, dataloader_valid


train_model()
