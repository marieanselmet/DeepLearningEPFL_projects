import math
import statistics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from nets import *


def train_model(model, auxiliary_loss, train_input, train_classes, train_target, mini_batch_size, learning_rate, opt):
    # number of epochs  
    nb_epochs = 25
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    
    # choice of optimizer 
    if (opt == 'SGD'):
        optimizer = optim.SGD(model.parameters(), learning_rate)
    if (opt == 'Adam'):
        optimizer = optim.Adam(model.parameters(), learning_rate)
    if (opt == 'RMSprop'):
        optimizer = optim.RMSprop(model.parameters(), learning_rate)
            
    for e in range(nb_epochs): 
        indices = torch.randperm(train_input.size(0))
        
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
        
        
def compute_nb_errors(model, data_input, data_classes, data_target, mini_batch_size):
    errors_img1 = 0
    errors_img2 = 0
    errors_total = 0
    
    for b in range(0, data_input.size(0), mini_batch_size):
        # calculate output
        img1, img2, _ = model(data_input.narrow(0, b, mini_batch_size))
        
        # predictions
        predicted_classes_img1 = img1.argmax(1)
        predicted_classes_img2 = img2.argmax(1)
        predicted_targets = (predicted_classes_img1 <= predicted_classes_img2).long()
        
        for k in range(mini_batch_size):
            # error for prediction of image 1
            if data_classes[b + k, 0] != predicted_classes_img1[k]:
                errors_img1 += 1
            # error for prediction of image 2
            if data_classes[b + k, 1] != predicted_classes_img2[k]:
                errors_img2 += 1
            # error for comparison of the images
            if data_target[b + k] != predicted_targets[k]:
                errors_total += 1
    
    return errors_img1, errors_img2, errors_total


def evaluate_model(train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, modelname, weight_sharing, auxiliary_loss, nb_hidden, optimizer, learning_rate, alpha = None):
    nb_tries = 10
    errors = []
    
    for i in range(nb_tries):
        # select the model
        if modelname == 'FNN':
            model = FNNNet(nb_hidden, weight_sharing)
        elif modelname == 'CNN':
            model = CNNNet(nb_hidden, weight_sharing)
        elif modelname == 'RELU':
            model = BestNet(nb_hidden, weight_sharing)
        elif modelname == 'LeakyRELU':
            model = LeakyBestNet(nb_hidden, weight_sharing)
        elif modelname == 'ELU':
            model = ELUBestNet(nb_hidden, alpha, weight_sharing)
        elif modelname == 'SELU':
            model = SELUBestNet(nb_hidden, weight_sharing)
    	# train and evaluate the model		
        train_model(model, auxiliary_loss, train_input, train_classes, train_target, mini_batch_size, learning_rate, optimizer)
        errors_img1, errors_img2, errors_total = compute_nb_errors(model, test_input, test_classes, test_target, mini_batch_size)
        errors.append(errors_total/10)
        print('Try {}   Digit recognition error: Img1 {}%  Img2 {}%   Digit comparison error {}%'.format(i+1, errors_img1/10, errors_img2/10, errors_total/10))
    
    errors_mean, errors_sd = round(statistics.mean(errors), 2), round(statistics.stdev(errors), 2)
    print('RESULT   Digit comparison error: Mean {}%  Standard deviation {}% \n'.format(errors_mean, errors_sd))
    return errors_mean, errors_sd


    
def find_best_lr(train_input, train_classes, train_target, mini_batch_size, weight_sharing, auxiliary_loss, nb_hidden, modelname, opt, alpha = None, plot = False):
    '''
    Returns the learning rate at which the loss is minimum, over 10 rounds.
    
    modelname : name of the chosen model, can be FNN, CNN, ReLU, LeakyReLU, ELU, SELU.
    opt : chosen optimizer, can be SGD, Adam or RMSprop.
    plot : boolean, if set to True the loss will be plotted as a function of the learning rate for each round. Set to False as default.
    '''
    nb_tries = 10
    nb_epochs = 25
    criterion = nn.CrossEntropyLoss()
    
    start_lr = 1e-5
    end_lr = 0.5
    # lambda function
    lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (nb_epochs * len(train_input)/mini_batch_size))
    
    opt_lrs = []
    for i in range(nb_tries):
        # choice of the model
        if modelname == 'FNN':
            model = FNNNet(nb_hidden, weight_sharing)
        elif modelname == 'CNN':
            model = CNNNet(nb_hidden, weight_sharing)
        elif modelname == 'RELU':
            model = BestNet(nb_hidden, weight_sharing)
        elif modelname == 'LeakyRELU':
            model = LeakyBestNet(nb_hidden, weight_sharing)
        elif modelname == 'ELU':
            model = ELUBestNet(nb_hidden, alpha, weight_sharing)
        elif modelname == 'SELU':
            model = SELUBestNet(nb_hidden, weight_sharing)
        # choice of the optimizer
        if (opt == 'SGD'):
            optimizer = optim.SGD(model.parameters(), start_lr)
        if (opt == 'Adam'):
            optimizer = optim.Adam(model.parameters(), start_lr)
        if (opt == 'RMSprop'):
            optimizer = optim.RMSprop(model.parameters(), start_lr)  
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        lr_find_loss = []
        lr_find_lr = []
        for e in range(nb_epochs):  
            # indices for batch
            indices = torch.randperm(train_input.size(0))
            # subsets for batch
            for b in range(0, train_input.size(0), mini_batch_size):
                indices_subset = indices[b:b+mini_batch_size]
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
                
                # update LR
                scheduler.step() 
                lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                lr_find_loss.append(loss)
         
        '''if plot:    
            # plot the loss as a function of the learning rate
            plt.figure()
            plt.ylabel("Loss")
            plt.xlabel("Learning rate [log]")
            plt.xscale("log")
            plt.plot(lr_find_lr, lr_find_loss)
            plt.show()'''
            
        # find index corresponding to the minimum loss
        index_min = min(range(len(lr_find_loss)), key=lr_find_loss.__getitem__)
        opt_lr = lr_find_lr[index_min]
        opt_lrs.append(opt_lr)
        
    best_lr = round(statistics.mean(opt_lrs), 4)
    print('Optimal learning rate: ', best_lr)
    
    return best_lr
