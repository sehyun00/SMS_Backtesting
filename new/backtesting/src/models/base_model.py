from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all models.
    Enforces a standard interface for training and inference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(
            config["project"]["device"] if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Standard PyTorch forward pass."""
        pass

    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference method.
        Should handle moving data to device and returning detached tensor (cpu).
        """
        pass

    def save(self, path: str):
        """Saves model state dict."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads model state dict."""
        self.load_state_dict(torch.load(path, map_location=self.device))
