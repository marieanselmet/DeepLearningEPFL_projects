import matplotlib.pyplot as plt
import torch
from dataHelpers import generate_pair_sets
from trainHelpers import evaluate_model, find_best_lr

# To prevent some conflict problems when running some plots on a MacOS (current issue with matplotlib)
# If not sufficient, set the KMP_DUPLICATE_LIB_OK variable of the virtual environment to true from the terminal
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
optimizer = 'SGD'
learning_rate = 1e-1
weight_sharing = True
auxiliary_loss = True


#****************************** Models error mean and standard deviation ******************************
ErrorsMean = []
ErrorsSD = []


#****************************** FNN model with weight sharing and an auxiliary loss ******************************
print("Evaluation of the FNN model with weight sharing and an auxiliary loss")
modelname = 'FNN'
mean, sd = evaluate_model(train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, modelname, weight_sharing, auxiliary_loss, nb_hidden, optimizer, learning_rate)
ErrorsMean.append(mean)
ErrorsSD.append(sd)


#****************************** LeNet-like CNN model with weight sharing and an auxiliary loss ******************************
print("Evaluation of the CNN model with weight sharing and an auxiliary loss")
modelname = 'CNN'
mean, sd = evaluate_model(train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, modelname, weight_sharing, auxiliary_loss, nb_hidden, optimizer, learning_rate)
ErrorsMean.append(mean)
ErrorsSD.append(sd)


#****************************** LeNet-like CNN model with weight sharing and an auxiliary loss + batch normalization ******************************
print("Evaluation of the CNN model with weight sharing and an auxiliary loss + batch normalization")
modelname = 'RELU'
mean, sd = evaluate_model(train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, modelname, weight_sharing, auxiliary_loss, nb_hidden, optimizer, learning_rate)
ErrorsMean.append(mean)
ErrorsSD.append(sd)


#****************************** FINAL BEST MODEL: LeNet-like CNN model with weight sharing and an auxiliary loss + batch normalization + LeakyReLU + optimal learning rate******************************
print("Evaluation of the CNN model with weight sharing and an auxiliary loss + batch normalization + LeakyReLU + optimal learning rate")
modelname = 'LeakyRELU'
best_lr = find_best_lr(train_input, train_classes, train_target, mini_batch_size, weight_sharing, auxiliary_loss, nb_hidden, modelname, optimizer)
mean, sd = evaluate_model(train_input, train_target, train_classes, test_input, test_target, test_classes, mini_batch_size, modelname, weight_sharing, auxiliary_loss, nb_hidden, optimizer, best_lr)
ErrorsMean.append(mean)
ErrorsSD.append(sd)


#****************************** Plot the models error mean and standard deviation ******************************
plt.figure()    
plt.errorbar(range(1, 5, 1), ErrorsMean, yerr=ErrorsSD, linestyle='--', fmt='-o',  elinewidth=1.5)
plt.xticks([1, 2, 3, 4], ["FNN", "CNN", "CNN + BN", "FINAL"])
plt.xlabel('Model')
plt.ylabel('Digit comparison error (%)')
plt.title('Mean and standard deviation of the digit comparison error for each model')
plt.show()
