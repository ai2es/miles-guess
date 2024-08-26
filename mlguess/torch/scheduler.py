import copy
import math
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


update_on_batch = ["cosine-annealing"]
update_on_epoch = ['lambda', 'plateau']


def load_scheduler(optimizer, conf):
    """Load a learning rate scheduler based on the configuration.

    Parameters:
        optimizer: The PyTorch optimizer.
        conf: The configuration dictionary.

    Returns:
        scheduler: The PyTorch learning rate scheduler.
    """
    conf = copy.deepcopy(conf)

    if conf['trainer']['use_scheduler']:
        scheduler_type = conf['trainer']['scheduler']['scheduler_type']
        del conf['trainer']['scheduler']['scheduler_type']
        if scheduler_type == 'lambda':
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_phase1)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, **conf['trainer']['scheduler'])
        elif scheduler_type == "cosine-annealing":
            scheduler = CosineAnnealingWarmupRestarts(optimizer, **conf['trainer']['scheduler'])
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}")
    else:
        scheduler = None

    return scheduler


# Define a half-cosine decay learning rate schedule for the second phase
def lr_lambda_phase2(step, total_updates_phase2=299000):
    """This function implements a half-cosine decay learning rate schedule
    specifically for the second training phase.

    Args:
        step (int): The current training step within the second phase.
        total_updates_phase2 (int, optional): The total number of updates
                                               in the second phase (default: 299000).

    Returns:
        torch.Tensor: The learning rate factor for the current step based on
                      the half-cosine decay schedule.
    """
    step_tensor = torch.tensor(step, dtype=torch.float32)
    return 0.5 * (1 + torch.cos((step_tensor / total_updates_phase2) * 3.1415))


# Combine the learning rate schedules
def phased_lr_lambda(step, total_updates_phase1=1000, total_updates_phase2=299000):
    """This function combines two learning rate schedules for a phased training
    process.

    Args:
        step (int): The current training step.
        total_updates_phase1 (int, optional): The total number of updates
                                                in the first phase (default: 1000).
        total_updates_phase2 (int, optional): The total number of updates
                                                in the second phase (default: 299000).

    Returns:
        torch.Tensor: The learning rate factor for the current step based on
                      which phase it belongs to and the corresponding schedule.
    """
    if step < total_updates_phase1:
        return lr_lambda_phase1(step, total_updates_phase1=total_updates_phase1)
    else:
        return lr_lambda_phase2(step - total_updates_phase1, total_updates_phase2=total_updates_phase2)


def lr_lambda_phase1(epoch, num_epochs=100, warmup_epochs=10):
    """This function implements a learning rate schedule based on the reference
    paper (https://arxiv.org/pdf/2312.03876.pdf) for the first training phase.

    Args:
        epoch (int): The current training epoch.
        num_epochs (int, optional): The total number of epochs in the training
                                     process (default: 100).
        warmup_epochs (int, optional): The number of epochs for the warmup
                                         phase (default: 10).

    Returns:
        float: The learning rate factor for the current epoch based on
               the specified learning rate schedule.
    """
    total_epochs = num_epochs - warmup_epochs

    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / total_epochs
        return 0.5 * (1 + math.cos(math.pi * progress))


class CosineAnnealingWarmupRestarts(LRScheduler):
    """
    Cosine Annealing learning rate scheduler.

    Attributes:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.,
        last_epoch: int = -1
        ):
        assert warmup_steps < first_cycle_steps
        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def annealed_probability(epoch, max_epochs=100, min_probability=0.01, max_probability=1.0):
    """Anneal the termination probability from 1 to a small value.

    Args:
        epoch: The current epoch.
        max_epochs: The maximum number of epochs for annealing.
        min_probability: The minimum termination probability.
        max_probability: The maximum termination probability.

    Returns:
        termination_probability: The annealed termination probability.
    """
    # Linear annealing schedule
    termination_probability = 1.0 - (epoch / max_epochs) * (1.0 - min_probability)

    # Ensure termination_probability is between min_probability and max_probability
    termination_probability = max(termination_probability, min_probability)
    termination_probability = min(termination_probability, max_probability)

    return termination_probability


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    num_epochs = 100
    batches_per_epoch = 500
    learning_rate = 1e-3

    model = torch.nn.Sequential(*[torch.nn.Linear(1, 2)])

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    # Assume optimizer is the optimizer you are using
    scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=batches_per_epoch,
            cycle_mult=6.0,
            max_lr=learning_rate,
            min_lr=1e-3 * learning_rate,
            warmup_steps=batches_per_epoch-1,
            gamma=0.7,
    )

    lr_values = []

    for epoch in range(num_epochs):
        for batch in range(batches_per_epoch):
            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate
            lr_values.append(optimizer.param_groups[0]['lr'])

    # Plot learning rate
    plt.plot(lr_values)
    plt.xlabel('Batch update')
    plt.ylabel('Learning rate')
    plt.show()
