import math

import torch

"""Exponential moving average model"""


class EMA:
    def __init__(self, beta, step_start):
        self.beta = beta
        self.step = 0
        self.step_start = step_start
        self.started = step_start <= 0

    @staticmethod
    def calculate_ema_factor(batch_size, dataset_size, max_epochs, ema_epochs = None):
        total_images = max_epochs * dataset_size
        if ema_epochs is None:
            ema_decay_time = round(0.05 * total_images) # EDM2 heuristic: 5% of total training images
        else:
            ema_decay_time = ema_epochs * dataset_size

        beta = math.exp(-batch_size / ema_decay_time)
        return beta

    @staticmethod
    def calculate_start_step(batch_size, dataset_size, start_epochs):
        """Convert an epoch offset to the number of optimizer steps."""
        return start_epochs * math.ceil(dataset_size / batch_size)

    def restore_state(self, completed_steps, started=None):
        """Restore progress while keeping the configured start step authoritative."""
        self.step = max(0, int(completed_steps))
        if self.step_start > 0 and self.step <= self.step_start:
            # This also handles checkpoints created with an earlier start
            # setting: increasing ema_start_epochs must delay EMA again.
            self.started = False
        elif started is None:
            # Legacy checkpoints did not store this flag. Their EMA weights
            # tracked the live weights before averaging, so they are a valid
            # initialization once the configured start has passed.
            self.started = True
        else:
            self.started = bool(started)

    def update_model_average(self, ema_model, model):
        pairs = [
            (ema_param, new_param)
            for ema_param, new_param in zip(ema_model.parameters(), model.parameters())
            if new_param.requires_grad
        ]
        if not pairs:
            return
        ema_params, new_params = zip(*pairs)
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, new_params, 1 - self.beta)

    def step_ema(self, ema_model, model):
        if self.step < self.step_start:
            self.step += 1
            return False

        if not self.started:
            self.reset_parameters(ema_model, model)
            self.started = True
        else:
            self.update_model_average(ema_model, model)
        self.step += 1
        return True

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
