import math
import torch
from torch import Tensor
torch.set_grad_enabled(False)



class Module(object):
    """ Module base class """

    def forward(self , * input):
        raise NotImplementedError

    def backward(self , * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

        
class Linear(Module):
    """ Implementation of a single fully connected linear layer module """

    def __init__(self, input_size, output_size):
        self.input = 0
        
        # Normal initialization of the weights (mean: 0, std: sqrt(Xavier variance))
        variance = 2/(input_size + output_size)
        self.weight = Tensor(output_size, input_size).normal_(0, math.sqrt(variance))
        self.weight_grad = Tensor(self.weight.size())

        # Normal initialization of the biases (mean: 0 mean, std: 1)
        self.bias = Tensor(output_size).normal_(0, math.sqrt(1))
        self.bias_grad = Tensor(self.bias.size())
        
        # For Momentum in SGD
        self.velocity_weight = torch.zeros_like(self.weight_grad)
        self.velocity_bias = torch.zeros_like(self.bias_grad)
        
        # For Adadelta
        self.square_avg_w = torch.zeros_like(self.weight_grad)
        self.square_avg_b = torch.zeros_like(self.bias_grad)

        self.delta_x_acc_w = torch.zeros_like(self.weight_grad)
        self.delta_x_acc_b = torch.zeros_like(self.bias_grad)

    def forward(self, input):
        self.input = input
        return self.input.mm(self.weight.t()) + self.bias

    def backward(self, output_grad):
        self.weight_grad.add_(output_grad.t().mm(self.input)) 
        self.bias_grad.add_(output_grad.sum(0))
        return output_grad.mm(self.weight) 

    def param(self):
        return [(self.weight, self.weight_grad, self.velocity_weight, self.square_avg_w, self.delta_x_acc_w),
                (self.bias, self.bias_grad, self.velocity_bias, self.square_avg_b, self.delta_x_acc_b)]


class Sequential(Module):
    """ Implementation of the sequential module """

    def __init__(self, modules):
        self.input = None
        self.modules = modules

    def forward(self, input):
        self.input = input
        output = self.input 
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, output_grad):
        grad_ = output_grad
        for m in reversed(self.modules):
            grad_ = m.backward(grad_)
        self.input = None
        return grad_

    def param(self):
        params = []
        for m in self.modules:
            params.extend(m.param())
        return params
