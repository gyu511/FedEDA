import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_curve, auc

import os
import copy
import time
import pickle
import wandb
import argparse
import parmap
import random
from datetime import datetime
import sys
import traceback

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from tqdm import tqdm

import datetime

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def my_collate(batch):
    x_batch = []
    y_batch = []
    for i, data_path in enumerate(batch):
        x, y = torch.load(data_path)
        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.cat(x_batch, dim=0)
    y_batch = torch.cat(y_batch, dim=0)
    return x_batch, y_batch

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details():
    print('\nExperimental details:')
    print(f'    Model     : {model}')
    print(f'    Optimizer : {optimizer}')
    print(f'    Learning  : {lr}')
    print(f'    Global Rounds   : {global_epoch}\n')

    print('    Federated parameters:')
    if iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {frac}')
    print(f'    Local Batch size   : {local_bs}')
    print(f'    Local Epochs       : {local_epoch}\n')
    return

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


#ResNet
class ResNet(nn.Module):
    def __init__(self, block, img_size, in_channel, out_channels, layers, fc_neurons, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = in_channel
        self.conv = conv3x3(in_channel, in_channel)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, out_channels[0], layers[0])
        self.layer2 = self.make_layer(block, out_channels[1], layers[1], 1)
        self.layer3 = self.make_layer(block, out_channels[2], layers[2], 1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        num_neurons = (img_size-2)**2 * out_channels[2]
        #self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        #num_neurons = img_size**2 * out_channels[2]
        self.fc = nn.Sequential(\
            nn.Linear(num_neurons, fc_neurons[0]),\
            nn.ReLU(inplace=True),\
            nn.Linear(fc_neurons[0], fc_neurons[1]),\
            nn.ReLU(inplace=True),\
            nn.Linear(fc_neurons[1], num_classes))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential( conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def main(args):
  # Use args to replace hardcoded values
  lr = args.lr
  device = args.device
  global_epoch = args.global_epoch
  local_epoch = args.local_epoch
  num_users = args.num_users
  imbalance = args.imbalance
  iid = args.iid
  batch_size = args.batch_size

  # Hyperparameters
  # Hyperparameters
  dataset = 'cad'
  #iid = 1
  out_channels = [60, 90, 120]
  in_channel = 46
  num_blocks = [ 2,3 ,3 ]
  num_neurons = [300 ,30 ]
  logger = SummaryWriter('../logs')

  criterion = nn.BCELoss()
  #lr = 0.01
  #batch_size = 2
  #lr = 0.01
 #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  hop = 15
  img_size = 2*hop+1
  #device = 'cuda:0'
  #global_epoch = 9
  #local_epoch = 10
  frac = 1
  #num_users = 3 
  threshold = 2
  #imbalance = 0

# argument : lr, device, global epoch, local epoch, num_users, imbalance, iid

  test_dir = "./x15/train"
  train_dir = "./x15/test"

  train_files = []
  test_files = []

  for file_name in os.listdir(train_dir):
      #print(file_name)
      train_files.append(os.path.join(train_dir, file_name))
  for file_name in os.listdir(test_dir):
      #print(file_name)
      test_files.append(os.path.join(test_dir, file_name))

  print('Train',len(train_files))
  print('Test',len(test_files))

  if imbalance ==0:
    num_items = int(len(train_files)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(train_files))]
    for i in range(num_users):
      dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
      all_idxs = list(set(all_idxs) - dict_users[i])
    user_groups = dict_users
  else:
    avg_items_per_user = int(len(train_files) / num_users)
    min_items_per_user = int(avg_items_per_user * 0.5)
    max_items_per_user = int(avg_items_per_user * 2)

    # Preallocate minimum samples to each user
    dict_users, all_idxs = {}, [i for i in range(len(train_files))]
    for i in range(num_users):
      dict_users[i] = set(np.random.choice(all_idxs, min_items_per_user, replace=False))
      all_idxs = list(set(all_idxs) - dict_users[i])
    
    # Distribute remaining samples
    while all_idxs:
      for i in range(num_users):
          # Skip distribution if there are no more samples to distribute
          if len(all_idxs) == 0:
            break
          
          if len(dict_users[i]) < max_items_per_user:
            # Maximum number of items that can be added to this user
            max_possible_extra = max_items_per_user - len(dict_users[i])
            # Maximum based on remaining unassigned items
            max_based_on_remainder = len(all_idxs)

            # Choose a random number in range [0, min(max_possible_extra, max_based_on_remainder)]
            extra_items = np.random.randint(0, min(max_possible_extra, max_based_on_remainder) + 1)

            if extra_items > 0:
              additional_samples = np.random.choice(all_idxs, extra_items, replace=False)
              dict_users[i].update(additional_samples)
              all_idxs = list(set(all_idxs) - set(additional_samples))
    user_groups = dict_users

  print('train_files',len(train_files))
  print('Batch Size: ', batch_size)
  # Display total data and number of users
  total_data = len(train_files)
  print(f"Total Data Items: {total_data}")
  print(f"Number of Users: {len(user_groups)}")

  # Display the distribution of data split for each user
  print("Data Distribution Among Users:")
  for user_id, user_indices in user_groups.items():
    print(f"User {user_id}: {len(user_indices)} items")

  # shuffle = False : Non-IID
  # shuffle = True : IID
  if iid:
    shuffle = True
  else:
    shuffle = False
  # Create a DataLoader for each user in the network
  user_loaders ={}
  for user_id, user_indices in user_groups.items():
    user_dataset = Subset(train_files, list(user_indices))
    user_loaders[user_id] = DataLoader(user_dataset, shuffle=shuffle, batch_size=batch_size, sampler=None, num_workers=2, collate_fn=my_collate)

  trainloader = DataLoader(train_files, shuffle=shuffle, batch_size=batch_size, sampler = None, num_workers=2, collate_fn = my_collate)
  testloader = DataLoader(test_files, batch_size=batch_size, sampler = None, num_workers=2, collate_fn = my_collate)

  # 모델 설정
  model = ResNet(ResidualBlock, img_size, in_channel, out_channels, num_blocks, num_neurons, 1)
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)


  # Set the model to train and send it to device.
  model.to(device)
  print('Device', device)
  model.train()
  print(model)

  # copy weights
  global_weights = model.state_dict()

  train_loss, train_accuracy = [], []
  print_every = 2

  # Global Round
  for epoch in (range(global_epoch)):
      local_weights, local_losses = [], []
      print(f'\n | Global Training Round : {global_epoch} |\n')

      model.train()
      m = max(int(frac * num_users), 1)
      idxs_users = np.random.choice(range(num_users), m, replace=False)
      user_progress = -1
      for idx in (idxs_users):
          user_progress = user_progress + 1
          user_loader = user_loaders[idx]
          total_user_data = len(user_loader.dataset)
          step = 0
          epoch_loss = []

          for iter in (range(local_epoch)):
              step += 1
              batch_loss = []
              #total_batches = len(user_loader)
              data_processed = 0
              for batch_idx, (images, labels) in enumerate(tqdm(user_loader, desc='Batch')):
                  #print('batch_idx', batch_idx)
                  x = images.squeeze().to(device)
                  #print('x', x.shape)
                  #print("Batch index:", batch_idx, "Batch size:", len(x))  # Debug print
                  y = labels.squeeze().to(device)
                  y = (y>threshold).float()
                  y_p = model(x)
                  y_p = torch.sigmoid(y_p)
                  #print('y_p', y_p)
                  loss = criterion(y_p.reshape((-1,1)), y.reshape((-1,1)))
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
                  
                  data_processed += batch_size
                  # Print every 100 batches
                  if (batch_idx + 1) % 100 == 0:
                    local_epoch_progress = (data_processed/total_user_data)*100
                    print(f'| Global Round : {epoch + 1}/{global_epoch} | Current User ID : {idx} | User Progress: {user_progress+1}/{num_users} | Local Epoch : {iter + 1}/{local_epoch} | Local Epoch Data Processed: {data_processed}/{total_user_data} ({local_epoch_progress:.0f}%) | Loss: {loss.item():.6f}')

                  #if (batch_idx % 100 == 0):
                    #print(f'| Global Round : {epoch} | User ID : {idx} | Local Epoch : {iter} | [{batch_idx}/{len(trainloader.dataset)/num_users} ({batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
                    #print('| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #      epoch,idx, iter, batch_idx * len(x)/ 100,
                    #      len(trainloader.dataset)/num_users,
                    #      100. * batch_idx / len(trainloader), loss.item()))

                  logger.add_scalar('loss', loss.item())
                  batch_loss.append(loss.item())

              epoch_loss.append(sum(batch_loss)/len(batch_loss))

          w = model.state_dict()
          loss = sum(epoch_loss) / len(epoch_loss)

          local_weights.append(copy.deepcopy(w))
          local_losses.append(copy.deepcopy(loss))

      # update global weights
      global_weights = average_weights(local_weights)

      # update global weights
      model.load_state_dict(global_weights)

      loss_avg = sum(local_losses) / len(local_losses)
      train_loss.append(loss_avg)

      # print global training loss after every 'i' rounds
      if (epoch+1) % print_every == 0:
          print(f' \nAvg Training Stats after {global_epoch+1} global rounds:')
          print(f'Training Loss : {np.mean(np.array(train_loss))}')
          #print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

      # Test Inference after Completion of training
      # test_acc, test_loss = test_inference(args, global_model, test_dataset)
  model.train(False)
  model.eval()
  test_loss, total, correct = 0.0, 0.0, 0.0

  test_loss = AverageMeter()
  trues = []
  preds = []
  batch_loss = []
  for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
      x = images.squeeze().to(device)
      y = labels.squeeze().to(device)
      y = (y>threshold).float()
      y_p = model(x)
      y_p = torch.sigmoid(y_p)
      batch_loss = criterion(y_p.reshape((-1,1)), y.reshape((-1,1)))

          # Inference
      test_loss.update(batch_loss.cpu().item(), y.shape[0])
     # test_loss += batch_loss.item()
         # Prediction
      _, pred_labels = torch.max(y_p, 1)
      pred_labels = pred_labels.view(-1)
      correct += torch.sum(torch.eq(pred_labels, y)).item()
      total += len(y)
      trues.extend(y.squeeze().cpu().tolist())
      preds.extend(y_p.squeeze().cpu().tolist())     
        

  test_acc = correct/total
  preds = np.array(preds, dtype=float)
  trues = np.array(trues, dtype=np.int64)
  fpr, tpr, th = roc_curve(trues, preds, pos_label=1)
  auroc = auc(fpr, tpr)
  preds = np.array(preds>0.5, dtype=np.int64)
  f1 = f1_score(trues, preds, pos_label=1, zero_division=0)
  recall = recall_score(trues, preds, pos_label=1, zero_division=0)
  precision = precision_score(trues, preds, pos_label=1, zero_division=0)
  acc = accuracy_score(trues, preds)
  log = "Epoch {epoch:d} Step {step:d} acc {acc:.3f} f1 {f1:.3f} auroc {auroc:.3f} recall {recall:.3f} precision {precision:.3f}"\
                .format(epoch=global_epoch, step=step, \
                        acc=acc, f1=f1, auroc=auroc, recall=recall, precision=precision)
  print(log)
  # Get current time for filename
  current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"metrics_batch_{batch_size}_lr_{lr}_global_{global_epoch}_local_{local_epoch}_users_{num_users}_imbalance_{imbalance}_iid_{iid}_{current_time}.txt"

  # Write metrics to the file
  with open(filename, 'w') as file:
    file.write(f"Accuracy: {acc}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"F1 Score: {f1}\n")

  print(f"Metrics saved in {filename}")



 # print(f' \n Results after {global_epoch} global rounds of training:')
 #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
 # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
      # Saving the objects train_loss and train_accuracy:
      #file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
      #    format(dataset, model, global_epoch, frac, iid,
      #            local_epoch, batch_size)

      #with open(file_name, 'wb') as f:
      #    pickle.dump([train_loss, train_accuracy], f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--global_epoch', type=int, default=9, help='Number of global epochs')
    parser.add_argument('--local_epoch', type=int, default=10, help='Number of local epochs')
    parser.add_argument('--num_users', type=int, default=3, help='Number of users')
    parser.add_argument('--imbalance', type=int, default=0, help='Imbalance in data distribution')
    parser.add_argument('--iid', type=int, default=1, help='IID setting')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')

    args = parser.parse_args()
    main(args)
