import statistics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from dataHelpers import generate_pair_sets
from nets import BestNet

# To prevent some conflict problems when running some plots on a MacOS (current issue with matplotlib)
# If not sufficient, set the KMP_DUPLICATE_LIB_OK variable of the virtual environment to true from the terminal
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_gradients_norm(nb_hidden, mini_batch_size, weight_sharing, auxiliary_loss, train_input, train_classes, train_target, learning_rate):
    '''
    Returns the gradient norm for each epoch and each layer of the BestNet model.
    '''
    nb_epochs = 25
    model = BestNet(nb_hidden, weight_sharing)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    
    Grad_conv1, Grad_conv2, Grad_fc1, Grad_fc2, Grad_fc_comp1, Grad_fc_comp2 = [], [], [], [], [], []
    for e in range(nb_epochs): 
        indices = torch.randperm(train_input.size(0))
        grad_conv1, grad_conv2, grad_fc1, grad_fc2, grad_fc_comp1, grad_fc_comp2 = [], [], [], [], [], []
        for b in range(0, train_input.size(0), mini_batch_size):
            # indices for batch
            indices_subset = indices[b:b+mini_batch_size]
            # subsets for batch
            train_input_subset = train_input.index_select(0, indices_subset)
            train_classes_subset = train_classes.index_select(0, indices_subset)
            train_target_subset = train_target.index_select(0, indices_subset)
            
            # calculate outputs
            img1, img2, both = model(train_input_subset)
            
            # calculate losses
            # loss of image 1
            loss_img1 = criterion(img1, train_classes_subset.narrow(1,0,1).view(-1)) 
            # loss of image 2
            loss_img2 = criterion(img2, train_classes_subset.narrow(1,1,1).view(-1))
            # loss for the comparison 
            loss_final = criterion(both, train_target_subset)
            loss = loss_final
            # auxiliary loss sums all losses
            if auxiliary_loss: loss += loss_img1 + loss_img2
            
            # zero gradients
            model.zero_grad()
            optimizer.zero_grad()
            
            # backward pass
            loss.backward()
            optimizer.step()
        
            grad_conv1.append(model.conv1.weight.grad.norm().item())
            grad_conv2.append(model.conv2.weight.grad.norm().item())
            grad_fc1.append(model.fc1.weight.grad.norm().item())
            grad_fc2.append(model.fc2.weight.grad.norm().item())
            grad_fc_comp1.append(model.fc_comp1.weight.grad.norm().item())
            grad_fc_comp2.append(model.fc_comp2.weight.grad.norm().item())
            
        Grad_conv1.append(statistics.mean(grad_conv1))
        Grad_conv2.append(statistics.mean(grad_conv2))
        Grad_fc1.append(statistics.mean(grad_fc1))
        Grad_fc2.append(statistics.mean(grad_fc2))
        Grad_fc_comp1.append(statistics.mean(grad_fc_comp1))
        Grad_fc_comp2.append(statistics.mean(grad_fc_comp2))
            
    return Grad_conv1, Grad_conv2, Grad_fc1, Grad_fc2, Grad_fc_comp1, Grad_fc_comp2


#****************************** Data ******************************
# Generate 1000 pairs of 14x14 size pictures
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

# Normalization of the inputs
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)


#****************************** Parameters ******************************
torch.seed()
nb_hidden = 100
mini_batch_size = 100
learning_rate = 1e-1
weight_sharing = True
auxiliary_loss = True


#****************************** Plot the evolution over the epochs of the norm of the gradient at each layer of the BestNet model. ******************************
grad_conv1, grad_conv2, grad_fc1, grad_fc2, grad_fc_comp1, grad_fc_comp2 = get_gradients_norm(nb_hidden, mini_batch_size, weight_sharing, auxiliary_loss, train_input, train_classes, train_target, learning_rate)

plt.figure()
epochs = range(25)
plt.plot(epochs, grad_conv1, label='1st layer')
plt.plot(epochs, grad_conv2, label='2nd layer')
plt.plot(epochs, grad_fc1, label='3rd layer')
plt.plot(epochs, grad_fc2, label='4th layer')
plt.plot(epochs, grad_fc_comp1, label='5th layer')
plt.plot(epochs, grad_fc_comp2, label='6th layer')
plt.xlabel('Epoch')
plt.ylabel('Gradient')
plt.legend()
plt.title('BestNet model')
plt.show()
