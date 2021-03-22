import math



class Scheduler():
    """ Implementation of the step decay and CLR schedulers """

    def __init__(self, lr):
        self.lr = lr
        
    def step_decay(self, epoch, initial_lr, drop = 0.5, epochs_drop = 10):
        self.lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return self.lr
    
    def cyclical_lr(self, nb_epochs, min_lr, max_lr, epoch):
        # Function to see where on the cycle we are
        def relative(it, stepsize):
            scaler = lambda x: 1/2
            cycle = math.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)
        
        # Lambda function to calculate the learning rate
        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, nb_epochs)
        self.lr = lr_lambda(epoch)
        return self.lr


