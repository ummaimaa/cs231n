import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    Inputs:
    - w: Current weights (numpy array)
    - dw: Current gradient (numpy array of same shape as w)
    - config: Dictionary with hyperparameters and velocity

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to vanilla SGD.
    - velocity: Numpy array of same shape as w, stores moving average of gradients.

    Returns:
    - next_w: Updated weights after one step of SGD with momentum
    - config: Updated config dictionary with new velocity
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    # Momentum update
    momentum = config["momentum"]
    lr = config["learning_rate"]

    # Compute new velocity
    v = momentum * v - lr * dw

    # Update weights using velocity
    next_w = w + v

    # Store velocity back into config
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    Inputs:
    - w: Current weights (numpy array)
    - dw: Current gradient (numpy array of same shape as w)
    - config: Dictionary with hyperparameters and cache

    config format:
    - learning_rate: Scalar learning rate
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared gradient cache
    - epsilon: Small scalar used for smoothing (to avoid dividing by zero)
    - cache: Moving average of second moments of gradients

    Returns:
    - next_w: Updated weights
    - config: Updated config with new cache
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    lr = config["learning_rate"]
    decay_rate = config["decay_rate"]
    eps = config["epsilon"]
    cache = config["cache"]

    # Update moving average of squared gradients
    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)

    # Update weights with element-wise scaling
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    # Store cache for next iteration
    config["cache"] = cache

    return next_w, config



def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    # Increment time step
    config['t'] += 1
    t = config['t']
    
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    lr = config['learning_rate']
    
    # Update biased first moment estimate
    config['m'] = beta1 * config['m'] + (1 - beta1) * dw
    # Update biased second raw moment estimate
    config['v'] = beta2 * config['v'] + (1 - beta2) * (dw ** 2)
    
    # Compute bias-corrected first and second moment estimates
    m_hat = config['m'] / (1 - beta1 ** t)
    v_hat = config['v'] / (1 - beta2 ** t)
    
    # Update weights
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    return next_w, config

    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

