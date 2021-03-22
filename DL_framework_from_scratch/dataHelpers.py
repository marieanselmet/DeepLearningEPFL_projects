import math
import torch


def target_to_one_hot(labels):
    """ Convert the labels to one-hot vectors for loss computation """
    nb_classes = labels.max() + 1
    targets = torch.empty(labels.size(0), nb_classes).zero_().scatter_(1, labels.view(-1, 1), 1)
    return targets


def generate_disc_set(nb=1000):
    """ Generate points uniformly sampled inside or outside a disk in [0,1]x[0,1], and the corresponding labels """
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = target_to_one_hot(((input-0.5).pow(2).sum(1).sub_(1 / (2*math.pi))).sign().sub(1).div(-2).long())
    return input, target


def normalize_data(train_input, validation_input, test_input):
    """ Normalize the train and test inputs """
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    validation_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)


def generate_data_set(nb):
    """ Generate the normalized dataset """
    train_input, train_target = generate_disc_set(nb)
    validation_input, validation_target = generate_disc_set(nb)
    test_input, test_target = generate_disc_set(nb)
    test_input_not_normalized = test_input.clone()
    normalize_data(train_input, validation_input, test_input)
    return train_input, train_target, validation_input, validation_target, test_input, test_target, test_input_not_normalized
