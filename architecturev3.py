import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
from dataloaderv6 import create_dataset_multithreaded
# from dataloaderv4 import create_dataset

import json
import os
import time
import cv2
import random
import numpy as np
import pickle

# Organize hyperparameters

batch_size = 16
frame_offset = 9

max_iters = 10000
context_pos = 'middle'
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
keypress_action_size = 23

img_scaling = 0.4
height = int(360 * img_scaling)
width = int(640 * img_scaling)
video_dir = 'labeller-training/video/'
actions_dir = 'labeller-training/actions/'


print(device)



class KeysConv3DNet(nn.Module):
    def __init__(self):
        super(KeysConv3DNet, self).__init__()
        
        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # Assuming input frames are of shape (360, 640)
        self.fc = nn.Linear(73728, keypress_action_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv3d1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv3d2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3d3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CameraConv3DNet(nn.Module):
    def __init__(self):
        super(CameraConv3DNet, self).__init__()

        self.conv3d1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.fc = nn.Linear(73728, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv3d1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv3d2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3d3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

keys_model = KeysConv3DNet().to(device)
camera_model = CameraConv3DNet().to(device)

keys_params = sum(p.numel() for p in keys_model.parameters())
camera_params = sum(p.numel() for p in camera_model.parameters())
print(f'total params {(keys_params + camera_params)/1e6}M ')
optimizer_keys = torch.optim.AdamW(keys_model.parameters(), lr=learning_rate)
optimizer_camera = torch.optim.AdamW(camera_model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # start_time = time.time()

    frames, keys_target, camera_target = create_dataset_multithreaded()
    # print(f'frames shape: {frames.shape}')
    

    frames = frames.unsqueeze(1)
    # print(f'dataloader time: {time.time()-start_time:.4f} \n')

    # start_inf_time = time.time()
    # print(frames.shape)    
    frames = frames.to(device).float() / 255.0  # Normalize pixel values
    keys_target = keys_target.to(device)
    camera_target = camera_target.to(device)
    
    # Forward pass
    keys_out = keys_model(frames)
    camera_out = camera_model(frames)
    
    loss_keysmodel = F.cross_entropy(keys_out, torch.max(keys_target, 1)[1])
    loss_cameramodel = F.mse_loss(camera_out, camera_target)

    # Backprop + loss
    optimizer_keys.zero_grad(set_to_none=True)
    optimizer_camera.zero_grad(set_to_none=True)
    
    loss_keysmodel.backward()
    loss_cameramodel.backward()

    # Gradient descent
    optimizer_keys.step()
    optimizer_camera.step()

    # end_inf_time = time.time()
    # print(f'inference time: {end_inf_time - start_inf_time:.4f}\n')
    # Loss reporting

    print(f'Iter {iter}, Keys Loss: {loss_keysmodel.item():.4f}, Camera Loss: {loss_cameramodel.item():.2f}\n')
    if iter % 100 == 0:
        
        with open('training_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Iter {iter}, Keys Loss: {loss_keysmodel.item():.4f}, Camera Loss: {loss_cameramodel.item():.2f}\n')
    # print(f'Iteration time: {time.time() - start_time:.4f} seconds\n\n')

with open('keys_model-0.4-imgscaling.pkl', 'wb') as f:
    pickle.dump(keys_model, f)

with open('camera_model-0.4-imgscaling.pkl', 'wb') as f:
    pickle.dump(camera_model, f)

print('Finished training')


# print('inference time:', time.time()-start_time)