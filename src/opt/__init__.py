# coding: utf-8
import attr
import torch.optim as optim
from .sam import SAM
from .kfac import KFACOptimizer
import math
import torch_optimizer
import pytorch_optimizer
from .shampoo import Shampoo
from .adahessian import AdaHessian


@attr.s
class OptimizerSetting:
    name = attr.ib()
    lr = attr.ib()
    weight_decay = attr.ib()
    model = attr.ib() #dict

    momentum = attr.ib(default=0.9) # sgd, sgd_nesterov
    eps = attr.ib(default=1e-8) # adam, rmsprop (term added to the denominator to improve numerical stability )

    beta_1 = attr.ib(default=0.9) #adam
    beta_2 = attr.ib(default=0.999) #adam

    rho = attr.ib(default=0.05) #rho

    damping = attr.ib(default=0.001) #kfac

    epsilon = attr.ib(default=1e-8) #shampoo
    update_freq = attr.ib(default=1) #shampoo

    hessian_power = attr.ib(default=1.0) #adahessian


def build_optimizer(setting: OptimizerSetting):
    name = setting.name
    model_params = setting.model.parameters()

    if name == 'vanilla_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        weight_decay=setting.weight_decay)

    elif name == 'momentum_sgd':
        return optim.SGD(params=model_params, 
                        lr=setting.lr, 
                        momentum=setting.momentum,
                        weight_decay=setting.weight_decay)
    elif name == 'adam':
        return optim.Adam(params=model_params, 
                        lr=setting.lr, 
                        betas=(setting.beta_1, setting.beta_2), 
                        eps=setting.eps, 
                        weight_decay=setting.weight_decay, 
                        amsgrad=False)
    elif name == 'sam':
        return SAM(params=model_params, 
                   base_optimizer=optim.SGD,
                   rho=setting.rho,
                   eps=setting.eps,
                   lr=setting.lr, 
                   weight_decay=setting.weight_decay, 
                   momentum=setting.momentum)
    
    elif name == 'quasinewton':
        return optim.LBFGS(params=model_params, 
                        lr=setting.lr)

    elif name == 'kfac':
        return KFACOptimizer(setting.model, 
                            lr=setting.lr, 
                            momentum=setting.momentum, 
                            damping=setting.damping, 
                            weight_decay=setting.weight_decay)

    elif name == 'adahessian':
        return AdaHessian(model_params,
                          lr=setting.lr,
                          weight_decay=setting.weight_decay,
                          hessian_power=setting.hessian_power)

    elif name == 'sophiah':
        return pytorch_optimizer.SophiaH(model_params,
                                         lr=setting.lr,
                                         p=setting.rho,
                                         weight_decay=setting.weight_decay)

    else:
        raise ValueError(
            'The selected optimizer is not supported for this trainer.')

      
def linear_warmup_cosine_decay(warmup_steps, total_steps):

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def linear_warmup_multistep_decay(warmup_steps, milestones, gamma=0.1):
    """
    Multi-step learning rate scheduler with linear warmup.
    
    Args:
        warmup_steps: Number of steps for linear warmup
        milestones: List of steps at which to decay the learning rate
        gamma: Multiplicative factor for learning rate decay (default: 0.1)
    """
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Count how many milestones have been passed
        num_decays = sum(1 for milestone in milestones if step >= milestone)
        return gamma ** num_decays
    
    return fn
