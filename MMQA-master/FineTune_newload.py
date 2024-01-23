from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np

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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import one_load
import two_load
import three_load
import mergenet

import csv
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True
ResultSave_path='TID2013_KADID_LIVEC.txt'

class ImageRatingsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):


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
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.sig(out)
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
            # labels = labels / 10.0
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

    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
    return sp, pl

def finetune_model():
    epochs = 35
    srocc_l = []
    plcc_l = []
    epoch_record = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('/home/user/MetaIQA-master/LIVE_WILD/')
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')
    images_fold = "/home/user/MetaIQA-master/LIVE_WILD/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(100):
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)

        print('\n')
        print('--------- The %2d rank trian-test (35epochs) ----------' % i )
        images_train, images_test = train_test_split(images, train_size = 0.8)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        #model = torch.load('model_IQA/TID2013_IQA_Meta_resnet18-1.pt')
        net_1 = one_load.densenetnew(pretrained=False)
        net_2 = two_load.densenetnew(pretrained=False)
        net_3 = three_load.densenetnew(pretrained=False)
        m_net = mergenet.merge_net(pretrained=False)
        l_net = BaselineModel1(1, 0.5, 1000)
        model = Net(net1 = net_1, net2 = net_2, net3 = net_3, mergenet = m_net, linear = l_net)

        model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_dense_newload.pt'))
        
        #model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()

        spearman = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)
            count = 0

            if epoch == 0:
                dataloader_valid1 = load_data('train1')
                dataloader_valid2 = load_data('train2')
                dataloader_valid3 = load_data('train3')
                model.eval()

                sp = computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model)[0]
                if sp > spearman:
                    spearman = sp
                print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')
            dataloader_train2 = load_data('train2')
            dataloader_train3 = load_data('train3')
            model.train()  # Set model to training mode
            running_loss = 0.0
            for data1, data2, data3 in zip(dataloader_train1, dataloader_train2, dataloader_train3):
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                #print('input1', inputs1)
                # labels = labels / 10.0
                inputs2 = data2['image']
                batch_size2 = inputs2.size()[0]
                #labels2 = data2['rating'].view(batch_size2, -1)
                #print('input2', inputs2)
                inputs3 = data3['image']
                batch_size3 = inputs3.size()[0]
                #labels3 = data3['rating'].view(batch_size3, -1)
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
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()
                
                #print('t  e  s  t %.8f' %loss.item())
                try:
                    running_loss += loss.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))
            with open('loss.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, epoch_loss])

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')
            dataloader_valid2 = load_data('test2')
            dataloader_valid3 = load_data('test3')
            model.eval()

            sp, pl = computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model)
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),'model_IQA/NEW_prior_model_multi.pt')

            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))

        srocc_l.append(spearman)
        plcc_l.append(plcc)
        
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print('PLCC: {:4f}, SROCC: {:4f}'.format(plcc, spearman),file=f1)
        
        with open('new_load_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, plcc, spearman])

    mean_srocc = sum(srocc_l)/len(srocc_l)
    mean_plcc = sum(plcc_l)/len(plcc_l)
    print('PLCC & SROCC', mean_srocc, mean_plcc)

    '''
    epoch_count = 0
    f = open('loss_record.txt','w')
    for line in epoch_record:
        epoch_record += 1
        f.write('epoch' + epoch_count + line + '\n')
        if epoch_record == 100:
            epoch_record = 0
    f.save()
    f.close()
    '''
    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train'):

    meta_num = 16
    data_dir = os.path.join('/home/user/MetaIQA-master/LIVE_WILD/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (224, 224)
    transformed_dataset_train1 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train2 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/salient_images/',
                                                    transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                  #RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train3 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/non_salient_images/',
                                                    transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                  #RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid1 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid2 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/salient_images/',
                                                    transform=transforms.Compose([Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid3 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/ChallengeDB_release/non_salient_images/',
                                                    transform=transforms.Compose([Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train1, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train2':
        dataloader = DataLoader(transformed_dataset_train2, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train3':
        dataloader = DataLoader(transformed_dataset_train3, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test1':
        dataloader = DataLoader(transformed_dataset_valid1, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test2':
        dataloader = DataLoader(transformed_dataset_valid2, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    else:
        dataloader = DataLoader(transformed_dataset_valid3, batch_size=bsize,
                                    shuffle=False, num_workers=4, collate_fn=my_collate)

    return dataloader

finetune_model()
