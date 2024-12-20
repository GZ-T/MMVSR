import torch
import cv2  
import glob
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 
import random
import numpy as np 
from PIL import Image


class train_GetData(Dataset):
    def __init__(self, Dir,batch1_index,batch2_index,batch3_index,batch4_index,Crop_size,crop_x,crop_y,flipud_flag,fliplr_flag,angle_flag,Is_Training=True):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch1_index = batch1_index
        self.batch2_index = batch2_index
        self.batch3_index = batch3_index
        self.batch4_index = batch4_index
        self.crop_size = Crop_size
        self.is_training = Is_Training 

        self.batch1_img_list = sorted(glob.glob(os.path.join(self.dir,'train_sharp', self.batch1_index, '*.png')))
        self.batch2_img_list = sorted(glob.glob(os.path.join(self.dir,'train_sharp', self.batch2_index, '*.png')))
        self.batch3_img_list = sorted(glob.glob(os.path.join(self.dir,'train_sharp', self.batch3_index, '*.png')))
        self.batch4_img_list = sorted(glob.glob(os.path.join(self.dir,'train_sharp', self.batch4_index, '*.png')))
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.flipud_flag = flipud_flag
        self.fliplr_flag = fliplr_flag
        self.angle_flag = angle_flag

        self.random_strat_frame = random.randint(0,85)

                     
    def __len__(self):

        return 15 # 2080Ti graphics memory can only support up to 15 frames of 64 * 64 (╯︵╰)
    
    def random_crop(self,hr,lr_x4,size,x,y):
        
        size_4 = size*4
        x_4, y_4 = x*4, y*4

        crop_lr_x4 = lr_x4[y:y+size, x:x+size].copy()
        crop_hr = hr[y_4:y_4+size_4, x_4:x_4+size_4].copy()

        return crop_hr, crop_lr_x4
    
    def random_flip_and_rotate(self,im1, im2,flipud_flag,fliplr_flag,angle_flag): 
        if flipud_flag < 0.5:
            im1 = np.flipud(im1)
            im2 = np.flipud(im2)

        if fliplr_flag < 0.5:
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)

        im1 = np.rot90(im1, angle_flag)
        im2 = np.rot90(im2, angle_flag)

        return im1.copy(), im2.copy()

    def __getitem__(self, index):
        hr_input1 = np.array(Image.open(self.batch1_img_list[self.random_strat_frame+index]))
        lr_input1 = np.array(Image.open(os.path.join(self.dir, 'train_sharp_bicubic/X4',self.batch1_index, self.batch1_img_list[self.random_strat_frame+index].split('/')[-1])))

        hr_input2 = np.array(Image.open(self.batch2_img_list[self.random_strat_frame+index]))
        lr_input2 = np.array(Image.open(os.path.join(self.dir, 'train_sharp_bicubic/X4',self.batch2_index, self.batch2_img_list[self.random_strat_frame+index].split('/')[-1])))

        hr_input3 = np.array(Image.open(self.batch3_img_list[self.random_strat_frame+index]))
        lr_input3 = np.array(Image.open(os.path.join(self.dir, 'train_sharp_bicubic/X4',self.batch3_index, self.batch3_img_list[self.random_strat_frame+index].split('/')[-1])))

        hr_input4 = np.array(Image.open(self.batch4_img_list[self.random_strat_frame+index]))
        lr_input4 = np.array(Image.open(os.path.join(self.dir, 'train_sharp_bicubic/X4',self.batch4_index, self.batch4_img_list[self.random_strat_frame+index].split('/')[-1])))


        hr_crop1,lr_crop1 = self.random_crop(hr_input1,lr_input1,self.crop_size,self.crop_x,self.crop_y)
        hr_crop2,lr_crop2 = self.random_crop(hr_input2,lr_input2,self.crop_size,self.crop_x,self.crop_y)
        hr_crop3,lr_crop3 = self.random_crop(hr_input3,lr_input3,self.crop_size,self.crop_x,self.crop_y)
        hr_crop4,lr_crop4 = self.random_crop(hr_input4,lr_input4,self.crop_size,self.crop_x,self.crop_y)
        
        hr_crop1,lr_crop1 = self.random_flip_and_rotate(hr_crop1 ,lr_crop1,self.flipud_flag,self.fliplr_flag,self.angle_flag)
        hr_crop2,lr_crop2 = self.random_flip_and_rotate(hr_crop2 ,lr_crop2,self.flipud_flag,self.fliplr_flag,self.angle_flag)
        hr_crop3,lr_crop3 = self.random_flip_and_rotate(hr_crop3 ,lr_crop3,self.flipud_flag,self.fliplr_flag,self.angle_flag)
        hr_crop4,lr_crop4 = self.random_flip_and_rotate(hr_crop4 ,lr_crop4,self.flipud_flag,self.fliplr_flag,self.angle_flag)
        
        
        hr_crop1 = self.transform(hr_crop1)
        lr_crop1 = self.transform(lr_crop1)
        hr_crop2 = self.transform(hr_crop2)
        lr_crop2 = self.transform(lr_crop2)
        hr_crop3 = self.transform(hr_crop3)
        lr_crop3 = self.transform(lr_crop3)
        hr_crop4 = self.transform(hr_crop4)
        lr_crop4 = self.transform(lr_crop4)
        
        return hr_crop1, lr_crop1, hr_crop2, lr_crop2, hr_crop3, lr_crop3, hr_crop4, lr_crop4
    
    
class val_GetData(Dataset):
    def __init__(self, Dir,batch1_index,Is_Training=True):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch1_index = batch1_index
        self.is_training = Is_Training 
        self.batch1_img_list = sorted(glob.glob(os.path.join(self.dir,'test_sharp', self.batch1_index, '*.png')))
                     
    def __len__(self):
        return len(self.batch1_img_list)
    
    def __getitem__(self, index):
        hr_input = np.array(Image.open(self.batch1_img_list[index]))
        lr_input = np.array(Image.open(os.path.join(self.dir, 'test_sharp_bicubic/X4',self.batch1_index, self.batch1_img_list[index].split('/')[-1])))
        
        hr_input = self.transform(hr_input)
        lr_input = self.transform(lr_input)
        
        return hr_input, lr_input
    





