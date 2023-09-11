"""
TODO:
  - add weight decay
"""

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        defaults={'lr':lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                p.data.add_(-group['lr']*p.grad.data)
        
        return loss


class Momentum(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.8, dampening=0.2):
        self.momentum = momentum
        self.dampening = dampening

        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
            
                if idx not in self.state['running_grad']:
                    self.state['running_grad'][idx] = p.grad.data
                else:
                    self.state['running_grad'][idx].add_(
                        self.state['running_grad'][idx].data * self.momentum + \
                        p.grad.data * (1 - self.dampening)
                    )
                
                p.data.add_(-group['lr'] * self.state['running_grad'][idx])


class Nestrov(torch.optim.Optimizer):
    """
    torch naive implementation:
    ----
    b_t = momentum * b_t-1 + g_t
    b_t = momentum * b_t + g_t
    w = w - lr * b_t

    another implementation:
    ----
    b_t = momentum * b_t-1 + dampening * g_t
    w = w - momentum * b_t - lr * g_t
    """
    def __init__(self, params, lr, momentum=0.8):
        self.momentum = momentum

        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):            
                if idx not in self.state['running_grad']:
                    self.state['running_grad'][idx] = p.grad.data
                else:
                    self.state['running_grad'][idx].add_(
                        self.state['running_grad'][idx].data * self.momentum + \
                        p.grad.data
                    )
                
                grad = self.momentum * self.state['running_grad'][idx] + p.grad.data
                p.data.add_(-group['lr'] * grad)


class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = {'lr': lr}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                # first moment
                if idx not in self.state['runing_1-th_moment']:
                    self.state['runing_1-th_moment'][idx] = p.grad.data
                else:
                    self.state['runing_1-th_moment'][idx] = self.beta1 * \
                        self.state['runing_1-th_moment'][idx] + (1 - self.beta1) * p.grad.data
                
                # second moment
                if idx not in self.state['runing_2-th_moment']:
                    self.state['runing_2-th_moment'][idx] = p.grad.data ** 2
                else:
                    self.state['runing_2-th_moment'][idx] = self.beta1 * \
                        self.state['runing_2-th_moment'][idx] + (1 - self.beta1) * p.grad.data ** 2
                
                first_moment = self.state['runing_1-th_moment'][idx] / (1 - self.beta1)
                second_moment = self.state['runing_2-th_moment'][idx] / (1-self.beta2)

                p.data.add_(-group['lr'] * (first_moment / (torch.sqrt(second_moment) + self.eps)))


class Nadam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = {'lr': lr}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                # first moment
                if idx not in self.state['runing_1-th_moment']:
                    self.state['runing_1-th_moment'][idx] = p.grad.data
                else:
                    self.state['runing_1-th_moment'][idx] = self.beta1 * \
                        self.state['runing_1-th_moment'][idx] + (1 - self.beta1) * p.grad.data
                
                # second moment
                if idx not in self.state['runing_2-th_moment']:
                    self.state['runing_2-th_moment'][idx] = p.grad.data ** 2
                else:
                    self.state['runing_2-th_moment'][idx] = self.beta1 * \
                        self.state['runing_2-th_moment'][idx] + (1 - self.beta1) * p.grad.data ** 2
                
                first_moment = self.beta1 * self.state['runing_1-th_moment'][idx] / (1 - self.beta1) + p.grad.data
                second_moment = self.state['runing_2-th_moment'][idx] / (1-self.beta2)

                p.data.add_(-group['lr'] * (first_moment / (torch.sqrt(second_moment) + self.eps)))


class Adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = {'lr': lr}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                p.data.add_(-group['lr'] * p.data)

                # first moment
                if idx not in self.state['runing_1-th_moment']:
                    self.state['runing_1-th_moment'][idx] = p.grad.data
                else:
                    self.state['runing_1-th_moment'][idx] = self.beta1 * \
                        self.state['runing_1-th_moment'][idx] + (1 - self.beta1) * p.grad.data
                
                # second moment
                if idx not in self.state['runing_2-th_moment']:
                    self.state['runing_2-th_moment'][idx] = p.grad.data ** 2
                else:
                    self.state['runing_2-th_moment'][idx] = self.beta1 * \
                        self.state['runing_2-th_moment'][idx] + (1 - self.beta1) * p.grad.data ** 2
                
                first_moment = self.state['runing_1-th_moment'][idx] / (1 - self.beta1)
                second_moment = self.state['runing_2-th_moment'][idx] / (1 - self.beta2)

                p.data.add_(-group['lr'] * (first_moment / (torch.sqrt(second_moment) + self.eps)))


