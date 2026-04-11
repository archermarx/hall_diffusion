"""Exponential moving average model"""
class EMA:
    def __init__(self, beta, step_start):
        self.beta = beta
        self.step = 0
        self.step_start = step_start

    def update_model_average(self, ema_model, model):
        for old_param, new_param in zip(ema_model.parameters(), model.parameters()):
            if not new_param.requires_grad:
                continue  # frozen params never change; skip the unnecessary GPU op
            old_weight, new_weight = old_param.data, new_param.data
            new_param.data = self.beta * old_weight + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model):
        if self.step < self.step_start:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())