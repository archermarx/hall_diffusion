"""Batched endpoint current measurements with analytic, sparse derivatives."""

import torch


ELEMENTARY_CHARGE = 1.602176634e-19  # C


class IonCurrentDensity:
    """Scalar ion current at the last grid point of a normalized model state."""

    def __init__(self, normalizer, channels, reference):
        names = [f"{quantity}_{charge}" for quantity in ("ni", "ui") for charge in range(1, 4)]
        metadata = [normalizer.find_name(name) for name in names]
        self.indices = torch.tensor(
            [channels[name] * reference.shape[-1] + reference.shape[-1] - 1 for name in names],
            device=reference.device,
        )
        self.means = reference.new_tensor([info["mean"][index] for index, info in metadata])
        self.stds = reference.new_tensor([info["std"][index] for index, info in metadata])
        self.logs = torch.tensor([info["log"][index] for index, info in metadata], device=reference.device)
        self.charges = reference.new_tensor([1, 2, 3]) * ELEMENTARY_CHARGE
        self.state_size = reference.numel()

    def _physical_values(self, states):
        values = states.flatten(start_dim=1).index_select(1, self.indices)
        values = self.means + self.stds * values
        # Exponentiate only log-transformed entries; linear velocities can be
        # large enough that exp(velocity) would overflow, even inside where().
        exponent = torch.where(self.logs, values, 0.0).exp()
        return torch.where(self.logs, exponent, values)

    def batch_apply(self, states):
        physical = self._physical_values(states)
        return (self.charges * physical[:, :3] * physical[:, 3:]).sum(dim=-1)

    @torch.no_grad()
    def derivatives(self, states):
        physical = self._physical_values(states)
        slopes = self.stds * torch.where(self.logs, physical, 1.0)
        return torch.cat((
            self.charges * physical[:, 3:] * slopes[:, :3],
            self.charges * physical[:, :3] * slopes[:, 3:],
        ), dim=-1)

    def __call__(self, state):
        return self.batch_apply(state.unsqueeze(0))[0]


class IonCurrentObservation:
    """One endpoint current plus linear rows, retaining their measurement order."""

    def __init__(self, rows):
        self.current_position, self.current = next(
            (index, row) for index, row in enumerate(rows) if isinstance(row, IonCurrentDensity)
        )
        linear_positions = [index for index, row in enumerate(rows) if isinstance(row, torch.Tensor)]
        self.linear_positions = self.current.indices.new_tensor(linear_positions)
        self.linear = (
            torch.stack([rows[index] for index in linear_positions])
            if linear_positions else self.current.means.new_empty((0, self.current.state_size))
        )
        self.output_order = self.current.indices.new_tensor(linear_positions + [self.current_position]).argsort()
        self.linear_endpoint_weights = self.linear.index_select(1, self.current.indices)
        self.linear_is_diagonal = bool(torch.all(torch.count_nonzero(self.linear, dim=0) <= 1).item())

    def combine(self, linear_values, current_values):
        values = torch.cat((linear_values, current_values.unsqueeze(-1)), dim=-1)
        return values.index_select(1, self.output_order)

    def batch_apply(self, states):
        linear_values = states.flatten(start_dim=1) @ self.linear.T
        return self.combine(linear_values, self.current.batch_apply(states))

    @torch.no_grad()
    def jacobians(self, states):
        result = states.new_zeros((states.shape[0], self.linear.shape[0] + 1, states[0].numel()))
        result[:, self.linear_positions] = self.linear
        result[:, self.current_position, self.current.indices] = self.current.derivatives(states)
        return result

    def __call__(self, state):
        return self.batch_apply(state.unsqueeze(0))[0]
