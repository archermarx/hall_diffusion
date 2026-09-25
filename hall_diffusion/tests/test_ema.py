import math

import torch

from hall_diffusion.models.ema import EMA


def set_weight(model, value):
    with torch.no_grad():
        model.weight.fill_(value)


def test_ema_start_epoch_is_converted_to_optimizer_steps():
    assert EMA.calculate_start_step(batch_size=3, dataset_size=5, start_epochs=4) == 8
    assert EMA.calculate_start_step(batch_size=3, dataset_size=5, start_epochs=1.5) == 3


def test_fractional_ema_epochs_preserve_example_based_decay():
    factor = EMA.calculate_ema_factor(
        batch_size=10,
        dataset_size=40,
        max_epochs=100.0,
        ema_epochs=6.25,
    )

    assert math.isclose(factor, math.exp(-10 / 250))


def test_ema_skips_prestart_copies_then_initializes_once_and_averages():
    model = torch.nn.Linear(1, 1, bias=False)
    ema_model = torch.nn.Linear(1, 1, bias=False)
    set_weight(model, 0.0)
    set_weight(ema_model, -1.0)
    ema = EMA(beta=0.5, step_start=2)

    set_weight(model, 1.0)
    assert ema.step_ema(ema_model, model) is False
    assert ema_model.weight.item() == -1.0
    set_weight(model, 2.0)
    assert ema.step_ema(ema_model, model) is False
    assert ema_model.weight.item() == -1.0

    set_weight(model, 3.0)
    assert ema.step_ema(ema_model, model) is True
    assert ema.started is True
    assert ema_model.weight.item() == 3.0

    set_weight(model, 5.0)
    assert ema.step_ema(ema_model, model) is True
    assert ema_model.weight.item() == 4.0


def test_resume_before_configured_start_does_not_enable_ema():
    steps_per_epoch = 10
    ema = EMA(beta=0.5, step_start=1024 * steps_per_epoch)

    # Simulate a legacy checkpoint made after ten epochs. Legacy checkpoints
    # have no saved `started` flag.
    ema.restore_state(completed_steps=10 * steps_per_epoch, started=None)

    assert ema.started is False
    assert ema.step_start == 1024 * steps_per_epoch


def test_current_config_can_delay_an_ema_that_an_old_config_started():
    ema = EMA(beta=0.5, step_start=1024)

    ema.restore_state(completed_steps=100, started=True)

    assert ema.started is False
    assert ema.step_start == 1024
