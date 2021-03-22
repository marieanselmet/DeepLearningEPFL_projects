import torch
from losscriteria import *
from optimizers import *
from schedulers import *
torch.set_grad_enabled(False)



def train_model(model, train_input, train_target, validation_input, validation_target, nb_epochs, mini_batch_size, learning_rate, momentum = 0, sched_ = None, opt = 'SGD', loss = 'MSE'):
    """ Train the model and return the losses over epochs """
    if opt == 'SGD' :
        optimizer = SGD(model.param(), learning_rate, momentum)
    elif opt == 'Adadelta':
        optimizer = Adadelta(model.param(), learning_rate)
    sched = Scheduler(learning_rate)
    
    if loss == 'MSE' :
        criterion = MSELoss()
    elif loss == 'CrossEntropy' :
        criterion = CrossEntropy()
    
    losses = []
    train_errors = []
    validation_errors = []

    for epoch in range(nb_epochs):
        acc_loss = 0
        nb_train_errors = 0
        indices = torch.randperm(train_input.size(0))
        
        for b in range(0, train_input.size(0), mini_batch_size):
            # indices for batch
            indices_subset = indices[b:b+mini_batch_size]
            # subsets for batch
            train_input_subset = train_input.index_select(0, indices_subset)
            train_target_subset = train_target.index_select(0, indices_subset)
            
            optimizer.zero_grad()                        
            output = model.forward(train_input_subset)
            
            for k in range(mini_batch_size):
                if torch.max(train_target.data[indices[b+k]], 0)[1] != torch.max(output[k], 0)[1]:
                    nb_train_errors += 1
            
            loss = criterion.forward(output, train_target_subset)
            acc_loss += loss
            
            output_grad = criterion.backward()
            model.backward(output_grad)
            optimizer.step()
        if sched_ == 'step_decay' :
            sched.step_decay(epoch, learning_rate, 0.5, nb_epochs/4)
        if sched_ == 'clr' :
            sched.cyclical_lr(nb_epochs, learning_rate/4, learning_rate, epoch)
        elif sched_ == None :
            pass
             
        losses.append(acc_loss)
        train_errors.append((100 * nb_train_errors) / train_input.size(0))
        
        nb_validation_errors, _ = compute_nb_errors(model, validation_input, validation_target, mini_batch_size)
        validation_errors.append((100 * nb_validation_errors) / validation_input.size(0))
        
        if epoch%10 == 0: print('Epoch {:d}   Train loss {:.02f}   Train error {:.02f}%   Validation error {:.02f}%'.format(epoch, acc_loss, (100 * nb_train_errors) / train_input.size(0), (100 * nb_validation_errors) / validation_input.size(0)))
   
    return losses, train_errors, validation_errors


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """ Compute the number of errors """
    nb_data_errors = 0
    misclassifications = torch.zeros(data_input.size(0),1)
    
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if torch.max(data_target.data[b + k], 0)[1] != torch.max(output[k], 0)[1]:
                nb_data_errors += 1
                misclassifications[b+k, 0] = 1
            else:
                misclassifications[b+k, 0] = 0
    return nb_data_errors, misclassifications


