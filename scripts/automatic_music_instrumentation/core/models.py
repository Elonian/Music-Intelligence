from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from scripts.automatic_music_instrumentation.core.data import MAX_DURATION_STEPS, N_CLASSES, TIME_STEPS_PER_BEAT


def init_note_model_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class EventFeatureEmbedding(nn.Module):
    """Embed note events as pitch + duration + beat + position."""

    def __init__(
        self,
        d_model: int,
        pitch_vocab: int = 128,
        duration_vocab: int = MAX_DURATION_STEPS + 1,
        beat_vocab: int = 64,
        position_vocab: int = TIME_STEPS_PER_BEAT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pitch_vocab = pitch_vocab
        self.duration_vocab = duration_vocab
        self.beat_vocab = beat_vocab
        self.position_vocab = position_vocab
        self.pitch_emb = nn.Embedding(pitch_vocab, d_model)
        self.duration_emb = nn.Embedding(duration_vocab, d_model)
        self.beat_emb = nn.Embedding(beat_vocab, d_model)
        self.position_emb = nn.Embedding(position_vocab, d_model)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_note_model_weights)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        events = events.long()
        pitch = events[:, :, 1].clamp(0, self.pitch_vocab - 1)
        duration = events[:, :, 2].clamp(0, self.duration_vocab - 1)
        beat = (events[:, :, 0] // TIME_STEPS_PER_BEAT).clamp(0, self.beat_vocab - 1)
        position = (events[:, :, 0] % TIME_STEPS_PER_BEAT).clamp(0, self.position_vocab - 1)
        scale = math.sqrt(self.d_model)
        embedded = (
            self.pitch_emb(pitch) * scale
            + self.duration_emb(duration) * scale
            + self.beat_emb(beat) * scale
            + self.position_emb(position) * scale
        )
        return self.dropout(embedded)


class BatchFirstPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for tensors shaped (batch, seq, dim)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs + self.pe[:, : inputs.size(1), :])


class TransformerPartSeparator(nn.Module):
    """Transformer part separator.

    With causal=False this is an offline encoder.
    With causal=True this is an online causal encoder.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
        beat_vocab: int = 64,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.embedding = EventFeatureEmbedding(d_model=d_model, beat_vocab=beat_vocab, dropout=dropout)
        self.pos_encoder = BatchFirstPositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_classes)
        self.fc_out.apply(init_note_model_weights)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, events: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.embedding(events)
        encoded = self.pos_encoder(encoded)
        mask = self._causal_mask(encoded.size(1), encoded.device)
        output = self.encoder(encoded, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)


class LSTMPartSeparator(nn.Module):
    """Online LSTM or offline BiLSTM part separator."""

    def __init__(
        self,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
        bidirectional: bool = False,
        beat_vocab: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = EventFeatureEmbedding(d_model=d_model, beat_vocab=beat_vocab, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(output_dim, n_classes)
        self.fc_out.apply(init_note_model_weights)

    def forward(self, events: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        del src_key_padding_mask
        embedded = self.embedding(events)
        output, _ = self.lstm(embedded)
        return self.fc_out(output)


class PerNoteMLPPartSeparator(nn.Module):
    """Independent per-note MLP baseline.

    This version keeps the independent-note behavior for event data.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
        beat_vocab: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = EventFeatureEmbedding(d_model=d_model, beat_vocab=beat_vocab, dropout=dropout)
        layers: list[nn.Module] = []
        input_dim = d_model
        for _ in range(num_layers):
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_size, n_classes))
        self.network = nn.Sequential(*layers)
        self.network.apply(init_note_model_weights)

    def forward(self, events: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        del src_key_padding_mask
        return self.network(self.embedding(events))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    online: bool


MODEL_SPECS = {
    "full_transformer": ModelSpec(
        name="full_transformer",
        description="Large offline encoder-only Transformer, d_model=512, layers=6, heads=8.",
        online=False,
    ),
    "compact_transformer": ModelSpec(
        name="compact_transformer",
        description="Compact offline encoder-only Transformer, d_model=128, layers=3, heads=8.",
        online=False,
    ),
    "causal_transformer": ModelSpec(
        name="causal_transformer",
        description="Online causal Transformer with masked self-attention.",
        online=True,
    ),
    "sequence_lstm": ModelSpec(
        name="sequence_lstm",
        description="Online 3-layer LSTM with 128 hidden units.",
        online=True,
    ),
    "bidirectional_lstm": ModelSpec(
        name="bidirectional_lstm",
        description="Offline 3-layer BiLSTM with 64 hidden units per direction.",
        online=False,
    ),
    "note_mlp": ModelSpec(
        name="note_mlp",
        description="Independent per-note MLP baseline.",
        online=False,
    ),
}


def build_model(model_name: str, n_classes: int = N_CLASSES) -> nn.Module:
    if model_name == "full_transformer":
        return TransformerPartSeparator(
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            n_classes=n_classes,
            beat_vocab=64,
            causal=False,
        )
    if model_name == "compact_transformer":
        return TransformerPartSeparator(
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            n_classes=n_classes,
            beat_vocab=64,
            causal=False,
        )
    if model_name == "causal_transformer":
        return TransformerPartSeparator(
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            n_classes=n_classes,
            beat_vocab=64,
            causal=True,
        )
    if model_name == "sequence_lstm":
        return LSTMPartSeparator(d_model=128, hidden_size=128, num_layers=3, n_classes=n_classes, bidirectional=False)
    if model_name == "bidirectional_lstm":
        return LSTMPartSeparator(d_model=128, hidden_size=64, num_layers=3, n_classes=n_classes, bidirectional=True)
    if model_name == "note_mlp":
        return PerNoteMLPPartSeparator(d_model=128, hidden_size=128, num_layers=3, n_classes=n_classes)
    raise ValueError(f"Unknown model name '{model_name}'. Options: {sorted(MODEL_SPECS)}")
