import numpy as np

def linear_decay(initial_lr, final_lr, epoch, total_epochs):
    lr = initial_lr - ((initial_lr - final_lr) / total_epochs) * epoch
    return max(lr, final_lr)

def power_law_decay(initial_lr, power, epoch, total_epochs):
    lr = initial_lr * (epoch / total_epochs) ** power
    return lr

def exponential_decay(initial_lr, decay_rate, epoch):
    lr = initial_lr * np.exp(-decay_rate * epoch)
    return lr

class LearningRateScheduler():
    def __init__(self, initial_lr, scheduler_type, **kwargs):
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.params = kwargs

    def get_lr(self, epoch, total_epochs):
        if self.scheduler_type == 'linear':
            final_lr = self.params.get('final_lr', 1e-6)
            return linear_decay(self.initial_lr, final_lr, epoch, total_epochs)
        elif self.scheduler_type == 'power':
            power = self.params.get('power', 1.0)
            return power_law_decay(self.initial_lr, power, epoch, total_epochs)
        elif self.scheduler_type == 'exponential':
            decay_rate = self.params.get('decay_rate', 0.01)
            return exponential_decay(self.initial_lr, decay_rate, epoch)
        else:
            raise ValueError(f"Scheduler type '{self.scheduler_type}' no reconocido.")
        