import torch
import torch.nn as nn

###############################################################################

class ResidualBlock(nn.Module):
      def __init__(self, channels_in, channels_out):
          super().__init__()
          self.ConvBlock = nn.Sequential(
              nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
              nn.BatchNorm2d(channels_out),
              nn.ReLU(),
              nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
              nn.BatchNorm2d(channels_out))
          self.Conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0)
          self.Relu = nn.ReLU()
      def forward(self, x):
          residual = x
          x = self.ConvBlock(x)
          x = x + self.Conv1(residual)
          x = self.Relu(x)
          return x

###############################################################################

class InceptionBlock(nn.Module):
      def __init__(self, channels_in, channels_out):
          super().__init__()
          self.ConvBlock1 = nn.Sequential(
              nn.Conv2d(int(channels_in/4), int(channels_out/4), kernel_size=1, padding=0),
              nn.Conv2d(int(channels_out/4), int(channels_out/4), kernel_size=3, padding=1),
              nn.Conv2d(int(channels_out/4), int(channels_out/4), kernel_size=3, padding=1))
          self.ConvBlock2 = nn.Sequential(
              nn.Conv2d(int(channels_in/4), int(channels_out/4), kernel_size=1, padding=0),
              nn.Conv2d(int(channels_out/4), int(channels_out/4), kernel_size=3, padding=1))
          self.ConvBlock3 = nn.Sequential(
              nn.Conv2d(int(channels_in/4), int(channels_out/4), kernel_size=1, padding=0))
          self.ConvBlock4 = nn.Sequential(
              nn.AvgPool2d(kernel_size=1, stride=1),
              nn.Conv2d(int(channels_in/4), int(channels_out/4), kernel_size=1, padding=0))
      def forward(self, x):
          xx = torch.chunk(x,4,dim=1)
          x1 = self.ConvBlock1(xx[0])
          x2 = self.ConvBlock2(xx[1])
          x3 = self.ConvBlock3(xx[2])
          x4 = self.ConvBlock4(xx[3])
          x = torch.cat((x1,x2,x3,x4),dim=1)
          return x

###############################################################################

class ResnextBlock(nn.Module):
      def __init__(self, channels_in, channels_out, group_size=32):
          super().__init__()
          self.ConvBlock = nn.Sequential(
              nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0),
              nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, groups=group_size))
          self.Conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0)
      def forward(self, x):
          residual = x
          x = self.ConvBlock(x)
          x = x + self.Conv1(residual)
          return x

###############################################################################

class BasicModel(nn.Module):
      def __init__(self, channels_in=3, channels_out=10):
          super().__init__()
          self.Block12 = nn.Sequential(
              nn.Conv2d(channels_in, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              nn.Conv2d(64, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.Block34_Residual = nn.Sequential(
              ResidualBlock(128,128),
              ResidualBlock(128,256),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.Block34_Inception = nn.Sequential(
              InceptionBlock(128,128),
              InceptionBlock(128,256),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.Block34_Resnext = nn.Sequential(
              ResnextBlock(128,128),
              ResnextBlock(128,256),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.Block5 = nn.Sequential(
              nn.Conv2d(256, 512, kernel_size=3, padding=1),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2))
          self.Block6_Residual = nn.Sequential(
              ResidualBlock(512,512),
              nn.AvgPool2d(kernel_size=4, stride=4))
          self.Block6_Inception = nn.Sequential(
              InceptionBlock(512,512),
              nn.AvgPool2d(kernel_size=4, stride=4))
          self.Block6_Resnext = nn.Sequential(
              ResnextBlock(512,512),
              nn.AvgPool2d(kernel_size=4, stride=4))
          self.Block78 = nn.Sequential(
              nn.Flatten(),
              nn.Linear(512, channels_out),
              nn.Softmax(dim=1))
      def forward(self, x):
          x = self.Block12(x)
          x = self.Block34_Residual(x)
          x = self.Block5(x)
          x = self.Block6_Residual(x)
          x = self.Block78(x)
          return x

###############################################################################

class InceptionModel(BasicModel):
      def __init__(self, channels_in=3, channels_out=10):
          super().__init__()
      def forward(self, x):
          x = self.Block12(x)
          x = self.Block34_Inception(x)
          x = self.Block5(x)
          x = self.Block6_Inception(x)
          x = self.Block78(x)
          return x

###############################################################################

class ResnextModel(BasicModel):
      def __init__(self, channels_in=3, channels_out=10):
          super().__init__()
      def forward(self, x):
          x = self.Block12(x)
          x = self.Block34_Resnext(x)
          x = self.Block5(x)
          x = self.Block6_Resnext(x)
          x = self.Block78(x)
          return x
