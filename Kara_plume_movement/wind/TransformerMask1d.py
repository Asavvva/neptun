import numpy as np
import torch
import torch.nn as nn


class ElementwiseInputMasking(nn.Module):
    """
    Поэлементное маскирование входа (B,T,F): случайно заменяет часть элементов на mask_value.

    p — доля замаскированных элементов (0..1).
    """
    def __init__(self, p: float = 0.15, mask_value: float = 0.0):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        self.p = p
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,T,F), got {tuple(x.shape)}")

        # mask=True -> замаскировать
        mask = (torch.rand_like(x) < self.p)
        mask_value = torch.as_tensor(self.mask_value, device=x.device, dtype=x.dtype)
        return torch.where(mask, mask_value, x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальные позиционные эмбеддинги для (B,T,D).
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


class SequenceToVectorTransformer(nn.Module):
    """
    Модель: вход (B,T,F) -> выход (B, out_dim) (вектор на всю последовательность).

    pooling: "mean" (по умолчанию) или "cls".
      - mean: среднее по времени (учитывая padding_mask, если он передан)
      - cls: добавляем [CLS]-токен и берем его выход
    """
    def __init__(
        self,
        in_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        out_dim: int = 128,
        mask_p: float = 0.15,
        mask_value: float = 0.0,
        max_len: int = 4096,
        pooling: str = "mean",  # "mean" | "cls"
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'.")

        self.pooling = pooling

        # 1) Маскирование значений входа (поэлементно)
        self.input_mask = ElementwiseInputMasking(p=mask_p, mask_value=mask_value)

        # 2) Проекция признаков -> d_model
        self.in_proj = nn.Linear(in_features, d_model)

        # 3) (опционально) CLS-токен
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # 4) Позиционные эмбеддинги
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        # 5) Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # ожидаем (B,T,D)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 6) Голова -> вектор
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        # init CLS
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B,T,F)
        padding_mask: (B,T) bool, где True = padding (игнорировать в attention и pooling)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,T,F), got {tuple(x.shape)}")

        # маскируем значения входа
        x = self.input_mask(x)

        h = self.in_proj(x)  # (B,T,D)

        if self.pooling == "cls":
            B = h.size(0)
            cls = self.cls_token.expand(B, 1, -1)  # (B,1,D)
            h = torch.cat([cls, h], dim=1)         # (B,T+1,D)

            if padding_mask is not None:
                # для CLS позиция не padding
                cls_pad = torch.zeros((B, 1), device=padding_mask.device, dtype=padding_mask.dtype)
                padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B,T+1)

        h = self.pos_enc(h)

        # TransformerEncoder: src_key_padding_mask=True -> игнорировать
        h = self.encoder(h, src_key_padding_mask=padding_mask)  # (B,T,D) или (B,T+1,D)

        if self.pooling == "cls":
            pooled = h[:, 0, :]  # (B,D)
        else:
            # mean pooling по непаддинговым шагам
            if padding_mask is None:
                pooled = h.mean(dim=1)
            else:
                valid = (~padding_mask).to(h.dtype)  # (B,T)
                denom = valid.sum(dim=1).clamp_min(1.0)  # (B,)
                pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

        return self.out(pooled)  # (B,out_dim)


# Пример:
if __name__ == "__main__":
    B, T, F = 8, 50, 32
    x = torch.randn(B, T, F)

    model = SequenceToVectorTransformer(
        in_features=F,
        d_model=128,
        nhead=4,
        num_layers=3,
        out_dim=64,
        mask_p=0.2,
        mask_value=0.0,
        pooling="mean",  # или "cls"
    )
    model.train()

    y = model(x)  # (B,64)
    print(y.shape)