import math
import matplotlib.pyplot as plt



def plot_targets(data_input, data_target, ax):
    circle = plt.Circle((0.5, 0.5), math.sqrt(1/(2*math.pi)), color='black', fill=False)
    ax.add_artist(circle)
    for n in range(data_input.size(0)):
        if (data_target[n, 0] == 0):
            ax.plot(data_input[n,0], data_input[n,1], 'go', markersize=3)
        else:
            ax.plot(data_input[n,0], data_input[n,1], 'ro', markersize=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    
def plot_targets_misclassifications(test_input_not_normalized, test_target, test_misclassified):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
    plot_targets(test_input_not_normalized, test_target, ax[0])
    plot_targets(test_input_not_normalized, test_misclassified, ax[1])
    fig.show()
    
    
def plot_loss_errors(nb_epochs, losses, train_errors, validation_errors):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
    ax[0].plot(range(1, nb_epochs+1), losses, 'g')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Train loss')
    ax[0].grid()
    
    ax[1].plot(range(1, nb_epochs+1), train_errors, 'g', label='Train')
    ax[1].plot(range(1, nb_epochs+1), validation_errors, 'r', label='Validation')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Error (%)')
    ax[1].legend()
    ax[1].grid()
    fig.show()