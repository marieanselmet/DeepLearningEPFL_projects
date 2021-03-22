import torch
from modules import Module
torch.set_grad_enabled(False)
    


class Sigmoid(Module):
    """ Implementation of the Sigmoid activation function """

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + torch.exp_(-self.input))
        return self.output

    def backward(self , output_grad):
        return output_grad * self.output * (1 - self.output)
    
    
class Tanh(Module):
    """ Implementation of the Tanh activation function """

    def forward(self, input):
        self.input = input
        self.output = self.input.tanh()
        return self.output
    
    def backward(self , output_grad):
        return output_grad * (1 - self.output ** 2)
    

class ReLU(Module):
    """ Implementation of the ReLU activation function """
    
    def forward(self, input):
        self.input = input
        self.output = torch.max(self.input,torch.zeros_like(self.input))
        return self.output

    def backward(self, output_grad):
        return output_grad * self.input.sign().clamp(min = 0.0)


class LeakyReLU(Module):
    """ Implementation of the LeakyReLU activation function """
    
    def __init__(self, leaky_slope):
        self.leaky_slope = leaky_slope
    
    def forward(self, input):
        self.input = input
        self.output = self.input.clamp(min = 0.0)   + (self.leaky_slope*self.input).clamp(max = 0.0)  
        return self.output

    def backward(self , output_grad):
        return output_grad * (self.input.sign().clamp(min = 0.0) + self.leaky_slope * ((-self.input.sign()).clamp(min = 0.0)))


class eLU(Module):
    """ Implementation of the eLU activation function """
    
    def _init_(self, alpha = 1.6732):
        self.alpha = alpha

    def forward(self, input, alpha = 1.6732):
        self.input = input
        a = alpha * (torch.exp(self.input) - 1)
        self.output = torch.max(self.input, torch.zeros_like(self.input)) + torch.min(a, torch.zeros_like(a))
        return self.output

    def backward(self, output_grad,alpha = 1.6732 ):
        a =  alpha * torch.exp(self.input)
        return output_grad * ((self.input.sign()).clamp(min = 0.0) + a * ((-self.input.sign()).clamp(min = 0.0)))
    
    
class SeLU(Module):
    """ Implementation of the SeLU activation function """
    
    def _init_(self, alpha = 1.6732, scale = 0.5):
        self.alpha = alpha
        self.scale = scale

    def forward(self, input, alpha = 1.6732, scale = 0.5):
        self.input = input
        a = alpha * (torch.exp(self.input) - 1)
        self.output = scale * (torch.max(self.input, torch.zeros_like(self.input)) + torch.min(a, torch.zeros_like(a)))
        return self.output

    def backward(self, output_grad, alpha = 1.6732, scale = 0.5):
        a =  alpha * torch.exp(self.input)
        return output_grad  *scale * ((self.input.sign()).clamp(min = 0.0) + a * ((-self.input.sign()).clamp(min = 0.0)))
    