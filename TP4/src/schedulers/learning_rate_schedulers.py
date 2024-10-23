import numpy as np

def linear_decay(initial_lr, final_lr, epoch, total_epochs):
    lr = initial_lr - ((initial_lr - final_lr) / total_epochs) * epoch
    return max(lr, final_lr)

def power_law_decay(initial_lr, power, epoch, total_epochs):
    lr = initial_lr * (1- (epoch / total_epochs)) ** power
    return max(lr, 0)

def exponential_decay(initial_lr, decay_rate, epoch):
    lr = initial_lr * np.exp(-decay_rate * epoch)
    return lr

class LearningRateScheduler():
    def __init__(self, initial_lr, scheduler_type=None, final_lr=1e-7, power=1, decay_rate=0.95):
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.final_lr = final_lr
        self.power = power
        self.decay_rate = decay_rate

    def get_lr(self, epoch, total_epochs):
        if self.scheduler_type == 'linear':
            return linear_decay(self.initial_lr, self.final_lr, epoch, total_epochs)
        elif self.scheduler_type == 'power':
            return power_law_decay(self.initial_lr, self.power, epoch, total_epochs)
        elif self.scheduler_type == 'exponential':
            return exponential_decay(self.initial_lr, self.decay_rate, epoch)
        else:
            return self.initial_lr
    
    def get_type(self):
        return self.scheduler_type