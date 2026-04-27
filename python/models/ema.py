import math

"""Exponential moving average model"""
class EMA:
    def __init__(self, beta, step_start):
        self.beta = beta
        self.step = 0
        self.step_start = step_start

    @staticmethod
    def calculate_ema_factor(batch_size, dataset_size, max_epochs, ema_epochs = None):
        total_images = max_epochs * dataset_size
        if ema_epochs is None:
            ema_decay_time = round(0.05 * total_images) # EDM2 heuristic: 5% of total training images
        else:
            ema_decay_time = ema_epochs * dataset_size

        beta = math.exp(-batch_size / ema_decay_time)
        return beta

    def update_model_average(self, ema_model, model):
        for ema_param, new_param in zip(ema_model.parameters(), model.parameters()):
            if not new_param.requires_grad:
                continue  # frozen params never change; skip the unnecessary GPU op
            ema_param.data = self.beta * ema_param.data + (1 - self.beta) * new_param.data

    def step_ema(self, ema_model, model):
        if self.step < self.step_start:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())