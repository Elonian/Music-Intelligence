from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from scripts.multitrack_generation.constants import FIELD_SPECS, TYPE_PAD


def init_multitrack_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.kaiming_uniform_(module.weight)


class MultitrackPositionalEncoding(nn.Module):
    """Sinusoidal encoding compatible with the multitrack notebook checkpoint.

    The notebook stores the buffer as (max_len, 1, d_model).  The default
    ``mode='sequence'`` uses it as a normal sequence-position encoding.  The
    ``mode='notebook'`` option reproduces the notebook forward pass exactly.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, mode: str = "sequence") -> None:
        super().__init__()
        if mode not in {"sequence", "notebook"}:
            raise ValueError("mode must be 'sequence' or 'notebook'")
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "notebook":
            encoded = inputs + self.pe[: inputs.size(0), :]
        else:
            encoded = inputs + self.pe[: inputs.size(1), :].transpose(0, 1)
        return self.dropout(encoded)


class MultitrackTransformer(nn.Module):
    """multitrack event Transformer with six output heads."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        positional_mode: str = "sequence",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.positional_mode = positional_mode

        self.type_emb = nn.Embedding(6, d_model)
        self.beat_emb = nn.Embedding(64, d_model)
        self.position_emb = nn.Embedding(24, d_model)
        self.pitch_emb = nn.Embedding(128, d_model)
        self.duration_emb = nn.Embedding(193, d_model)
        self.instrument_emb = nn.Embedding(5, d_model)

        self.pos_encoder = MultitrackPositionalEncoding(d_model, dropout, mode=positional_mode)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out_type = nn.Linear(d_model, 6)
        self.fc_out_beat = nn.Linear(d_model, 64)
        self.fc_out_position = nn.Linear(d_model, 24)
        self.fc_out_pitch = nn.Linear(d_model, 128)
        self.fc_out_duration = nn.Linear(d_model, 193)
        self.fc_out_instrument = nn.Linear(d_model, 5)
        self.apply(init_multitrack_weights)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        events: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        events = events.long()
        if src_mask is None:
            src_mask = self._causal_mask(events.shape[1], events.device)
        if src_key_padding_mask is None:
            src_key_padding_mask = events[:, :, 0] == TYPE_PAD

        scale = math.sqrt(self.d_model)
        encoded = (
            self.type_emb(events[:, :, 0].clamp(0, 5)) * scale
            + self.beat_emb(events[:, :, 1].clamp(0, 63)) * scale
            + self.position_emb(events[:, :, 2].clamp(0, 23)) * scale
            + self.pitch_emb(events[:, :, 3].clamp(0, 127)) * scale
            + self.duration_emb(events[:, :, 4].clamp(0, 192)) * scale
            + self.instrument_emb(events[:, :, 5].clamp(0, 4)) * scale
        )
        encoded = self.pos_encoder(encoded)
        hidden = self.encoder(encoded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return [
            self.fc_out_type(hidden),
            self.fc_out_beat(hidden),
            self.fc_out_position(hidden),
            self.fc_out_pitch(hidden),
            self.fc_out_duration(hidden),
            self.fc_out_instrument(hidden),
        ]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    description: str


MODEL_SPECS = {
    "full": ModelSpec("full", 512, 8, 6, 2048, "Notebook-sized Multitrack Transformer."),
    "compact": ModelSpec("compact", 128, 8, 3, 256, "Faster local Multitrack Transformer for smoke tests and iteration."),
    "tiny": ModelSpec("tiny", 64, 4, 2, 128, "Very small model for CPU smoke tests."),
}


def build_model(model_name: str = "full", dropout: float = 0.1, positional_mode: str = "sequence") -> MultitrackTransformer:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model '{model_name}', expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model_name]
    return MultitrackTransformer(
        d_model=spec.d_model,
        nhead=spec.nhead,
        num_layers=spec.num_layers,
        dim_feedforward=spec.dim_feedforward,
        dropout=dropout,
        positional_mode=positional_mode,
    )


def _clean_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def infer_model_name_from_state_dict(state: dict[str, torch.Tensor]) -> str:
    d_model = int(state["type_emb.weight"].shape[1])
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("encoder.layers.") and key.split(".")[2].isdigit()
    }
    num_layers = max(layer_ids) + 1 if layer_ids else 0
    for name, spec in MODEL_SPECS.items():
        if spec.d_model == d_model and spec.num_layers == num_layers:
            return name
    return "full" if d_model == 512 else "compact"


def checkpoint_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        return _clean_state_dict_keys(payload["model_state"])
    if isinstance(payload, dict) and "state_dict" in payload:
        return _clean_state_dict_keys(payload["state_dict"])
    if isinstance(payload, dict):
        tensor_items = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
        if tensor_items:
            return _clean_state_dict_keys(tensor_items)
    raise ValueError("checkpoint does not contain a model state dict")


def load_model_checkpoint(
    checkpoint_path: Path | str,
    model_name: str | None = None,
    device: torch.device | str | None = None,
    positional_mode: str | None = None,
    strict: bool = True,
) -> MultitrackTransformer:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(Path(checkpoint_path), map_location=resolved_device)
    state = checkpoint_state_dict(payload)
    payload_config = payload.get("config") if isinstance(payload, dict) and isinstance(payload.get("config"), dict) else {}
    if model_name is None and isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        model_name = str(payload["config"].get("model_name") or payload["config"].get("model") or "")
    if not model_name:
        model_name = infer_model_name_from_state_dict(state)
    if positional_mode is None:
        positional_mode = str(payload_config.get("positional_mode") or ("notebook" if not payload_config else "sequence"))
    model = build_model(model_name=model_name, positional_mode=positional_mode).to(resolved_device)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def output_vocab_sizes() -> list[int]:
    return [field.vocab_size for field in FIELD_SPECS]
