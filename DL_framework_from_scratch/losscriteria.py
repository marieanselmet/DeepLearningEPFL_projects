from modules import Module



class MSELoss(Module):
    """ Implementation of the MSE loss criterion """

    def forward(self, input, target):
        self.input = input
        self.target = target
        return (self.input - self.target).pow(2).mean()

    def backward(self):
        return 2 * (self.input - self.target).div(self.input.size(0)) 
    

class CrossEntropy(Module):
    """ Implementation of the Cross Entropy loss criterion """
       
    def forward(self, input, target):
        softmax = input.exp().div(input.exp().sum(1).view(-1,1))
        self.input = softmax
        self.target = target
        return -(softmax.log()*target).mean()

    def backward(self):
        return (self.input - self.target).div(self.input.size(0))
