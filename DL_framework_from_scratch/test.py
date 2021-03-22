import statistics
import torch
from dataHelpers import generate_data_set
from trainHelpers import train_model, compute_nb_errors
from modules import Linear, Sequential
from plots import *
from activations import *


# To prevent some conflict problems when running some plots on a MacOS (current issue with matplotlib)
# If not sufficient, set the KMP_DUPLICATE_LIB_OK variable of the virtual environment to true from the terminal
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




#****************************** Data ******************************
nb_samples = 1000
train_input, train_target, validation_input, validation_target, test_input, test_target, test_input_not_normalized = generate_data_set(nb_samples)


#****************************** Parameters ******************************
torch.manual_seed(12)
nb_tries = 5
nb_hidden = 25
nb_epochs = 400
mini_batch_size = 100
learning_rate = 1e-2

#****************************** Training and testing of the model ******************************
test_errors = []

for n in range(nb_tries):
    print("Try: "+str(n+1))

    modules = [
                Linear(2, nb_hidden),
                eLU(),
                Linear(nb_hidden, nb_hidden),
                ReLU(),
                Linear(nb_hidden, nb_hidden),
                LeakyReLU(0.01),
                Linear(nb_hidden, 2),
                Tanh()
                ]
    model = Sequential(modules)
    losses, train_errors, validation_errors = train_model(model, train_input, train_target, validation_input, validation_target, nb_epochs, mini_batch_size, learning_rate, 0, None, 'Adadelta', 'MSE')
    nb_test_errors, test_misclassified = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('Test error: {:0.2f}% ({:d}/{:d})'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    test_errors.append((100 * nb_test_errors) / test_input.size(0))
    
    
    # Plots
    #plot_loss_errors(nb_epochs, losses, train_errors, validation_errors)
    #plot_targets_misclassifications(test_input_not_normalized, test_target, test_misclassified)


# Mean and standard deviation of the model test errors over all the tries
test_errors_mean, test_errors_sd = statistics.mean(test_errors), statistics.stdev(test_errors)
print('\n RESULT   Test error over all tries: Mean {:0.2f}%  Standard deviation {:0.2f}% \n'.format(test_errors_mean, test_errors_sd))
