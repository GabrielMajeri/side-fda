import torch

def euclidean_loss(y, y_hat):
    """Unbalanced Euclidean loss which has a bigger error value as the absolute
    depth of the scene increases."""
    diffs = y_hat - y
    total = torch.sum(diffs ** 2)

    return total / (2 * len(y))

def balance_function(y):
    """Balancing function which tries to ensure the scale of the error is
    the same for both far and near depths.
    """

    # Two constants parameterizing this function:
    # - The first one should be a relatively large positive number
    # - The second one should be a small negative number
    a1, a2 = 1.5, -0.1

    return a1 * y + (a2 / 2) * (y ** 2)

def depth_balanced_euclidean_loss(y, y_hat):
    """Loss function which uses a balancing function to ensure the network
    learns to estimate near depths as well as far ones."""

    return euclidean_loss(balance_function(y), balance_function(y_hat))
