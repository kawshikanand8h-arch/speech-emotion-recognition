import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.features import FEATURE_DIM


class EmotionBiLSTM(nn.Module):
    """Bidirectional LSTM with attention + mean pooling over time."""

    def __init__(
        self,
        input_size=FEATURE_DIM,
        hidden_size=128,
        num_layers=2,
        num_classes=7,
        dropout=0.4,
        input_dropout=0.1,
        layer_norm=False,
        bottleneck=0,
    ):
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_size = hidden_size * 2
        # Optional: normalise LSTM outputs (stabilises training, fewer
        # degrees of freedom for memorising speakers).
        self.layer_norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(out_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        if bottleneck > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(out_size * 2, bottleneck),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(out_size * 2, num_classes),
            )

    def forward(self, x, lengths):
        x = self.input_dropout(x)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )  # (batch, time, 2*hidden)
        outputs = self.layer_norm(outputs)

        steps = torch.arange(outputs.size(1), device=outputs.device)
        mask = steps[None, :] < lengths.to(outputs.device)[:, None]  # (B, T)

        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention_pooled = (weights * outputs).sum(dim=1)

        mask_f = mask.unsqueeze(-1).float()
        mean_pooled = (outputs * mask_f).sum(dim=1) / mask_f.sum(dim=1)

        pooled = torch.cat([attention_pooled, mean_pooled], dim=1)
        return self.classifier(pooled)


def build_model(config):
    return EmotionBiLSTM(**config)
