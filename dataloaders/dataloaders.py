import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

###############################################################################

def DataLoader_Torch(ROOT='./datasets'):
    
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
    
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
    
    train_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transforms_train)
    val_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms_val)
    
    return train_dataset, val_dataset

###############################################################################
    
def DataLoader_Manual(DIR_TRAIN, DIR_VAL):

    classes = os.listdir(DIR_TRAIN)
    train_imgs = [] ; val_imgs  = []
    for _class in classes:
        train_imgs += glob.glob(DIR_TRAIN + _class + '/*.jpg')
        val_imgs += glob.glob(DIR_VAL + _class + '/*.jpg')
        
    print("Total Classes:", len(classes))
    print("Total train images:", len(train_imgs))
    print("Total test images:", len(val_imgs))

    class MyDataset:
        
      def __init__(self, imgs_list, classes, transforms=None):
          super(MyDataset, self).__init__()
          self.imgs_list = imgs_list
          self.class_to_int = {classes[i] : i for i in range(len(classes))}
          self.transforms = transforms
          
      def __getitem__(self, index):
          image_path = self.imgs_list[index]
          # Reading image
          image = Image.open(image_path)
          # Retriving class label
          label = image_path.split("/")[-2]
          label = self.class_to_int[label]
          # Applying transforms on image
          if self.transforms is not None:
              image = self.transforms(image)
          else:
              image = transforms.ToTensor()(image)
          return image, label
          
      def __len__(self):
          return len(self.imgs_list)
    
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
    
    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

    train_dataset = MyDataset(imgs_list = train_imgs, classes = classes, transforms = transforms_train)
    val_dataset = MyDataset(imgs_list = val_imgs, classes = classes, transforms = transforms_val)
    
    return train_dataset, val_dataset

###############################################################################

def PlotSample(batch_data, mean, std, to_denormalize = False, figsize = (1,1)):

    batch_image,_ = batch_data
    batch_size = batch_image.shape[0]
    
    random_batch_index = random.randint(0,batch_size-1)
    random_image = batch_image[random_batch_index]
    
    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    
    if to_denormalize:
        image_transposed = np.array(std)*image_transposed + np.array(mean)
        image_transposed = image_transposed.clip(0,1)
        
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_transposed)
    ax.set_axis_off()