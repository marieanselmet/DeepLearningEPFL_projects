import torch
from torch import nn
from torch.nn import functional as F


class FNNNet(nn.Module):
    def __init__(self, nb_hidden, weight_sharing = False):
        """
        Fully connected neural network with three linear layers for the digit recognition task and two linear layers for the digit comparison task.
        ReLU is used as activation function.
        
        nb_hidden : number of hidden layers.
        weight_sharing : boolean, set to False if not specified.
        """
        super(FNNNet, self).__init__()
        self.weight_sharing = weight_sharing
        # linear layers for the digit recognition task
        self.fc1 = nn.Linear(196, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)  
        self.fc3 = nn.Linear(nb_hidden, 10) 
        if not weight_sharing:
            self.fc1_ = nn.Linear(196, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, nb_hidden)  
            self.fc3_ = nn.Linear(nb_hidden, 10)
        # linear layers for the digit comparison task
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
                             
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1).view(-1, 196)
        img2 = input.narrow(1, 1, 1).view(-1, 196)
        # architecture of the neural network and activation functions
        x = F.relu(self.fc1(img1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # weight sharing
        if self.weight_sharing:
            y = F.relu(self.fc1(img2))
            y = F.relu(self.fc2(y))
            y = self.fc3(y)
        else: 
            y = F.relu(self.fc1_(img2))
            y = F.relu(self.fc2_(y))
            y = self.fc3_(y)
        # concatenation
        xy = torch.cat((x,y), 1)
        z = F.relu(self.fc_comp1(xy))
        z = self.fc_comp2(z)
        return x, y, z


class CNNNet(nn.Module):   
    def __init__(self, nb_hidden, weight_sharing = False):
        """
        Convolutional neural network based on LeNet model architecture.
        ReLU is used as activation function.
        
        nb_hidden : number of hidden layers 
        weight_sharing : boolean, set to False if not specified. 
        """
        super(CNNNet, self).__init__()
        self.weight_sharing = weight_sharing
        # linear and convolutional layers for the digit recognition task
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        if not weight_sharing:
            self.conv1_ = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2_ = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1_ = nn.Linear(256, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, 10)
        # linear layers for the digit comparison task
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
                             
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1)
        img2 = input.narrow(1, 1, 1)
        # architecture of the neural network
        x = F.relu(F.max_pool2d(self.conv1(img1), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        # weight sharing
        if self.weight_sharing:
            y = F.relu(F.max_pool2d(self.conv1(img2), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
            y = F.relu(self.fc1(y.view(-1, 256)))
            y = self.fc2(y)
        else:
            y = F.relu(F.max_pool2d(self.conv1_(img2), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.conv2_(y), kernel_size=2, stride=2))
            y = F.relu(self.fc1_(y.view(-1, 256)))
            y = self.fc2_(y)
        # concatenation
        xy = torch.cat((x,y), 1)
        z = F.relu(self.fc_comp1(xy))
        z = self.fc_comp2(z)
        return x, y, z
    
    
class BestNet(nn.Module):   
    def __init__(self, nb_hidden, weight_sharing = False):
        """
        Convolutional neural network based on LeNet model architecture.
        Batch normalization after each of the two convolutional layers.
        ReLU is used as activation function.
        
        nb_hidden : number of hidden layers 
        weight_sharing : boolean, set to False if not specified.
        """
        super(BestNet, self).__init__()
        self.weight_sharing = weight_sharing
        # convolutional and linear layers with batch normalization for the digit recognition task
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        if not weight_sharing:
            self.conv1_ = nn.Conv2d(1, 32, kernel_size=3)
            self.bn1_ = nn.BatchNorm2d(32)
            self.conv2_ = nn.Conv2d(32, 64, kernel_size=3)
            self.bn2_ = nn.BatchNorm2d(64)
            self.fc1_ = nn.Linear(256, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, 10)    
        # linear layers for the digit comparison task
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
                             
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1)
        img2 = input.narrow(1, 1, 1)
        # architecture of the neural network with activation function ReLU
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)       
        # weight sharing
        if self.weight_sharing:
            y = F.relu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2))
            y = F.relu(self.fc1(y.view(-1, 256)))
            y = self.fc2(y)
        else:
            y = F.relu(F.max_pool2d(self.bn1_(self.conv1_(img2)), kernel_size=2, stride=2))
            y = F.relu(F.max_pool2d(self.bn2_(self.conv2_(y)), kernel_size=2, stride=2))
            y = F.relu(self.fc1_(y.view(-1, 256)))
            y = self.fc2_(y)
        # concatenation
        xy = torch.cat((x,y), 1)
        z = F.relu(self.fc_comp1(xy))
        z = self.fc_comp2(z)
        return x, y, z
    
    
class LeakyBestNet(nn.Module):   
    def __init__(self, nb_hidden, weight_sharing = False):
        """
        Convolutional neural network based on LeNet model architecture.
        Batch normalization after each of the two convolutional layers.
        LeakyReLU is used as activation function, with the default value alpha=0.01.
        
        nb_hidden : number of hidden layers 
        weight_sharing : boolean, set to False if not specified.
        """
        super(LeakyBestNet, self).__init__()
        self.weight_sharing = weight_sharing
        # convolutional and linear layers with batch normalization for the digit recognition task
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        if not weight_sharing:
            self.conv1_ = nn.Conv2d(1, 32, kernel_size=3)
            self.bn1_ = nn.BatchNorm2d(32)
            self.conv2_ = nn.Conv2d(32, 64, kernel_size=3)
            self.bn2_ = nn.BatchNorm2d(64)
            self.fc1_ = nn.Linear(256, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, 10)    
        # linear layers for the digit comparison task
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
                             
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1)
        img2 = input.narrow(1, 1, 1)
        # architecture of the neural network with activation function ReLU
        x = F.leaky_relu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2))
        x = F.leaky_relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.leaky_relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)       
        # weight sharing
        if self.weight_sharing:
            y = F.leaky_relu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2))
            y = F.leaky_relu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2))
            y = F.leaky_relu(self.fc1(y.view(-1, 256)))
            y = self.fc2(y)
        else:
            y = F.leaky_relu(F.max_pool2d(self.bn1_(self.conv1_(img2)), kernel_size=2, stride=2))
            y = F.leaky_relu(F.max_pool2d(self.bn2_(self.conv2_(y)), kernel_size=2, stride=2))
            y = F.leaky_relu(self.fc1_(y.view(-1, 256)))
            y = self.fc2_(y)
        # concatenation
        xy = torch.cat((x,y), 1)
        z = F.relu(self.fc_comp1(xy))
        z = self.fc_comp2(z)
        return x, y, z


class ELUBestNet(nn.Module):   
    def __init__(self, nb_hidden, alpha = 1, weight_sharing = False):
        """
        Convolutional neural network based on LeNet model architecture.
        Batch normalization after each of the two convolutional layers.
        ELU is used as activation function.
        
        nb_hidden : number of hidden layers 
        alpha : is a parameter for ELU. It controls  the  scale  of  the  small exponential  increase of x when x < 0. Set to 1 as default.
        weight_sharing : boolean, set to False if not specified
        """
        super(ELUBestNet, self).__init__()
        self.alpha = alpha
        self.weight_sharing = weight_sharing
        # convolutional and linear layers with batch normalization for the digit recognition task
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10) 
        if not weight_sharing:
            self.conv1_ = nn.Conv2d(1, 32, kernel_size=3)
            self.bn1_ = nn.BatchNorm2d(32)
            self.conv2_ = nn.Conv2d(32, 64, kernel_size=3)
            self.bn2_ = nn.BatchNorm2d(64)
            self.fc1_ = nn.Linear(256, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, 10)
        # linear layers for the digit comparison task  
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
                              
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1)
        img2 = input.narrow(1, 1, 1)
        # architecture of the neural network with activation function ELU
        x = F.elu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2), self.alpha)
        x = F.elu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2), self.alpha)
        x = F.elu(self.fc1(x.view(-1, 256)), self.alpha)
        x = self.fc2(x)
        # weight sharing
        if self.weight_sharing:
            y = F.elu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2), self.alpha)
            y = F.elu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2), self.alpha)
            y = F.elu(self.fc1(y.view(-1, 256)), self.alpha)
            y = self.fc2(y)
        else:
            y = F.elu(F.max_pool2d(self.bn1_(self.conv1_(img2)), kernel_size=2, stride=2), self.alpha)
            y = F.elu(F.max_pool2d(self.bn2_(self.conv2_(y)), kernel_size=2, stride=2), self.alpha)
            y = F.elu(self.fc1_(y.view(-1, 256)), self.alpha)
            y = self.fc2_(y)
        #concatenation
        xy = torch.cat((x,y), 1)
        z = F.elu(self.fc_comp1(xy), self.alpha)
        z = self.fc_comp2(z)
        
        return x, y, z
    
    
class SELUBestNet(nn.Module):   
    def __init__(self, nb_hidden, weight_sharing = False):
        """
        Convolutional neural network based on LeNet model architecture.
        Batch normalization after each of the two convolutional layers.
        SELU is used as activation function.
        
        nb_hidden : number of hidden layers 
        weight_sharing : boolean, set to False if not specified
        """
        super(SELUBestNet, self).__init__()
        self.weight_sharing = weight_sharing
        # convolutional and linear layers with batch normalization for the digit recognition task
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10) 
        # weight sharing
        if not weight_sharing:
            self.conv1_ = nn.Conv2d(1, 32, kernel_size=3)
            self.bn1_ = nn.BatchNorm2d(32)
            self.conv2_ = nn.Conv2d(32, 64, kernel_size=3)
            self.bn2_ = nn.BatchNorm2d(64)
            self.fc1_ = nn.Linear(256, nb_hidden)
            self.fc2_ = nn.Linear(nb_hidden, 10)
        # linear layers for the digit comparison task       
        self.fc_comp1 = nn.Linear(20, 10)
        self.fc_comp2 = nn.Linear(10, 2)
        
    def selu(x, alpha= 1.6732, scale = 1.0507):
        '''
        Definition of the SELU activation function, which is a scaled version of ELU with a specific alpha. 
        
        alpha : parameter for the activation function ELU. It controls  the  scale  of  the  small exponential  increase of x when x < 0. Set to 1.6732 for the SELU function.
        scale : scale factor for the ELU function, set as 1.0507 for the SELU function.
        '''
        return scale * F.elu(x, alpha)
                              
    def forward(self, input):
        # separate images from input
        img1 = input.narrow(1, 0, 1)
        img2 = input.narrow(1, 1, 1)
        # architecture of the neural network with activation function SELU
        x = F.selu(F.max_pool2d(self.bn1(self.conv1(img1)), kernel_size=2, stride=2))
        x = F.selu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.selu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        # weight sharing
        if self.weight_sharing:
            y = F.selu(F.max_pool2d(self.bn1(self.conv1(img2)), kernel_size=2, stride=2))
            y = F.selu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2))
            y = F.selu(self.fc1(y.view(-1, 256)))
            y = self.fc2(y)
        else:
            y = F.selu(F.max_pool2d(self.bn1_(self.conv1_(img2)), kernel_size=2, stride=2))
            y = F.selu(F.max_pool2d(self.bn2_(self.conv2_(y)), kernel_size=2, stride=2))
            y = F.selu(self.fc1_(y.view(-1, 256)))
            y = self.fc2_(y)
        #concatenation
        xy = torch.cat((x,y), 1)
        z = F.selu(self.fc_comp1(xy))
        z = self.fc_comp2(z)
        return x, y, z
