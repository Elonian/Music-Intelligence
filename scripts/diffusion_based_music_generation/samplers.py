from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.diffusion_based_music_generation.model import NULL_PITCH


def _time_batch(batch_size: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), float(value), device=device)


def guided_velocity(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    pitches: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Return conditional velocity, optionally with classifier-free guidance."""
    v_cond = model(x, t, pitches)
    if float(guidance_scale) == 1.0:
        return v_cond
    null_pitches = torch.full_like(pitches, NULL_PITCH)
    v_uncond = model(x, t, null_pitches)
    return v_uncond + float(guidance_scale) * (v_cond - v_uncond)


def euler_sample(model, x1: torch.Tensor, pitches: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
    """Euler ODE integration from noise at t=1 to data at t=0."""
    dt = 1.0 / int(n_steps)
    x = x1.clone()
    batch_size = x.shape[0]
    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t_batch = _time_batch(batch_size, t_value, x.device)
            v = model(x, t_batch, pitches)
            x = x - v * dt
    return x


def naive_scale_sample(
    model,
    x1: torch.Tensor,
    pitches: torch.Tensor,
    n_steps: int = 50,
    scale: float = 1.0,
) -> torch.Tensor:
    """Euler sampling with a scalar multiplier on every velocity prediction."""
    dt = 1.0 / int(n_steps)
    x = x1.clone()
    batch_size = x.shape[0]
    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t_batch = _time_batch(batch_size, t_value, x.device)
            v = model(x, t_batch, pitches) * float(scale)
            x = x - v * dt
    return x


def cfg_sample(
    model,
    x1: torch.Tensor,
    pitches: torch.Tensor,
    n_steps: int = 50,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Euler sampling with classifier-free guidance."""
    dt = 1.0 / int(n_steps)
    x = x1.clone()
    batch_size = x.shape[0]
    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t_batch = _time_batch(batch_size, t_value, x.device)
            v = guided_velocity(model, x, t_batch, pitches, guidance_scale)
            x = x - v * dt
    return x


def heun_sample(
    model,
    x1: torch.Tensor,
    pitches: torch.Tensor,
    n_steps: int = 50,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Second-order Heun integration from t=1 to t=0, with optional CFG."""
    dt = 1.0 / int(n_steps)
    x = x1.clone()
    batch_size = x.shape[0]
    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t_batch = _time_batch(batch_size, t_value, x.device)
            k1 = guided_velocity(model, x, t_batch, pitches, guidance_scale)
            x_predict = x - k1 * dt
            t_next = _time_batch(batch_size, max(t_value - dt, 0.0), x.device)
            k2 = guided_velocity(model, x_predict, t_next, pitches, guidance_scale)
            x = x - 0.5 * (k1 + k2) * dt
    return x


def rk4_sample(
    model,
    x1: torch.Tensor,
    pitches: torch.Tensor,
    n_steps: int = 25,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Classic fourth-order Runge-Kutta integration from t=1 to t=0."""
    dt = 1.0 / int(n_steps)
    x = x1.clone()
    batch_size = x.shape[0]
    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t1 = _time_batch(batch_size, t_value, x.device)
            t_mid = _time_batch(batch_size, max(t_value - 0.5 * dt, 0.0), x.device)
            t_next = _time_batch(batch_size, max(t_value - dt, 0.0), x.device)
            k1 = guided_velocity(model, x, t1, pitches, guidance_scale)
            k2 = guided_velocity(model, x - 0.5 * dt * k1, t_mid, pitches, guidance_scale)
            k3 = guided_velocity(model, x - 0.5 * dt * k2, t_mid, pitches, guidance_scale)
            k4 = guided_velocity(model, x - dt * k3, t_next, pitches, guidance_scale)
            x = x - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def sample_timesteps(B: int, device, t_sample: str = "logit_normal") -> torch.Tensor:
    """Sample timesteps in [0, 1] for flow matching training."""
    if t_sample == "uniform":
        return torch.rand(int(B), device=device)
    if t_sample == "logit_normal":
        return torch.sigmoid(torch.randn(int(B), device=device))
    raise ValueError("t_sample must be 'uniform' or 'logit_normal'")


def flow_loss(
    model,
    x_data: torch.Tensor,
    pitch: torch.Tensor,
    t: torch.Tensor,
    p_uncond: float = 0.1,
) -> torch.Tensor:
    """Differentiable flow matching loss with classifier-free dropout."""
    noise = torch.randn_like(x_data)
    t_view = t[:, None, None, None]
    x_t = (1.0 - t_view) * x_data + t_view * noise
    target = noise - x_data

    pitch_input = pitch.clone()
    if p_uncond > 0:
        mask = torch.rand(pitch_input.shape[0], device=x_data.device) < float(p_uncond)
        pitch_input[mask] = NULL_PITCH

    v_pred = model(x_t, t, pitch_input)
    return F.mse_loss(v_pred, target)


SAMPLERS = {
    "euler": euler_sample,
    "cfg": cfg_sample,
    "heun": heun_sample,
    "rk4": rk4_sample,
    "naive": naive_scale_sample,
}
