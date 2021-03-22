
class Optimizer(object):
    """ Optimizer base class """

    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):    
        raise NotImplementedError
        
        
class SGD(Optimizer):
    """ Implementation of the SGD optimizer """
    
    def __init__(self, params, lr, momentum = 0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
    
    def step(self):
        """Single optimization step """
        for p, grad, v, square_grad_avg, delta_x_acc in self.params:
            v = self.momentum * v + self.lr*grad
            p.add_(-v)
    
    def zero_grad(self):
        """Clears the gradients in all the modules parameters"""
        for p, grad, v, square_grad_avg, delta_x_acc in self.params:
            grad.zero_()


class Adadelta(Optimizer): 
    """ Implementation of the ADADELTA optimizer """

    def __init__(self, params, lr, rho=0.9, eps=1e-6):
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps 
 
    def step(self):
        """ Performs a single optimization step """
        for p, grad, v, square_grad_avg, delta_x_acc in self.params:
            # Compute the running average of the squared gradients 
            square_grad_avg.mul_(self.rho)
            square_grad_avg.addcmul_(grad, grad, value = 1 - self.rho)
            # Compute the RMS of the previous squared gradients (eps to avoid numerical issues later for division)
            std = (square_grad_avg.add_(self.eps)).sqrt_()
            # Compute the accumulated update
            delta_x = ((delta_x_acc.add_(self.eps)).sqrt_()) * grad / std
            # Accumulate the updates
            delta_x_acc.mul_(self.rho)
            delta_x_acc.addcmul_(delta_x, delta_x, value = 1 - self.rho) 
            # Update the parameters
            p.add_(delta_x, alpha = - self.lr)

    def zero_grad(self):
        """Clears the gradients in all the modules parameters"""
        for p, grad, v, square_grad_avg, delta_x_acc in self.params:
            grad.zero_()
