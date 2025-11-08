# -*- coding: utf-8 -*-
"""
간단 씬 분류기 (menu / battle / field)
- 매우 경량 CNN
- PIL.Image 입력 → torch.Tensor 전처리 → 확률 반환

환경 변수:
- HERO4_STATE_MODEL : 가중치 경로 (기본 models/state_cnn.pt)
- HERO4_USE_CLASSIFIER : 1이면 사용 시도
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from PIL import Image
import numpy as np


CLASS_NAMES = ["menu", "battle", "field"]


class _StateCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 20 * 15, 128)  # 입력 160x120 기준
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 160x120 -> 80x60
        x = self.pool(F.relu(self.conv2(x)))   # 80x60 -> 40x30
        x = self.pool(F.relu(self.conv3(x)))   # 40x30 -> 20x15
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


@dataclass
class StatePrediction:
    label: Optional[str]
    probs: Dict[str, float]
    confidence: float


class StateClassifier:
    def __init__(self, model_path: str = None, device: str = None):
        if torch is None:
            raise RuntimeError("PyTorch 가 필요합니다 (requirements.txt 내 torch)")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _StateCNN(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model_path = model_path
        self._last_mtime: Optional[float] = None
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            try:
                self._last_mtime = os.path.getmtime(model_path)
            except Exception:
                self._last_mtime = None
        self.model.eval()
        self.size = (160, 120)
    @staticmethod
    def from_env() -> Optional["StateClassifier"]:
        if torch is None:
            return None
        use = os.environ.get("HERO4_USE_CLASSIFIER", "1")
        if str(use) not in ("1", "true", "True"):
            return None
        path = os.environ.get("HERO4_STATE_MODEL", os.path.join("models", "state_cnn.pt"))
        # 가중치가 없어도 로드 가능(무작위 초기화) → 신뢰도 낮음, 훈련 전까지는 휴리스틱 참고용
        return StateClassifier(model_path=path)
    def maybe_reload(self) -> bool:
        """모델 파일 변경 시 핫-리로드. 성공 시 True 반환."""
        if not self.model_path:
            return False
        try:
            mtime = os.path.getmtime(self.model_path)
        except Exception:
            return False
        if self._last_mtime is None or mtime > self._last_mtime:
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self._last_mtime = mtime
                return True
            except Exception:
                return False
        return False
    def _preprocess(self, img: Image.Image) -> "torch.Tensor":
        # PIL RGB -> 160x120 -> Tensor(C,H,W), [0,1]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(self.size)
        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t
    @torch.no_grad()
    def predict(self, img: Image.Image) -> StatePrediction:
        x = self._preprocess(img)
        logits = self.model(x)[0].detach().cpu()
        probs = torch.softmax(logits, dim=0).numpy().tolist()
        idx = int(np.argmax(probs))
        conf = float(max(probs))
        prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        return StatePrediction(CLASS_NAMES[idx], prob_map, conf)

__all__ = ["StateClassifier", "StatePrediction", "CLASS_NAMES"]
