"""
Base backend class for DLRM inference.

Defines the abstract interface that all DLRM inference backends must implement.
"""

from abc import ABC, abstractmethod


class DLRMBackend(ABC):
    """
    Abstract base class for DLRM inference backends.

    Provides a common interface for different backend implementations,
    allowing flexibility in model architecture and optimization strategies.

    Attributes:
        model_name: Name identifier for the model.
        model_impl: Concrete model implementation (set by subclasses).
    """

    def __init__(self, model_name: str):
        """
        Initialize the DLRM backend.

        Args:
            model_name: Name identifier for the model.
        """
        self.model_name = model_name
        self.model_impl = None

    @abstractmethod
    def initialize(self):
        """
        Initialize the backend with model configuration and weights.

        Must be implemented by subclasses to set up the model for inference.
        """
        raise NotImplementedError("DLRMBackend:initialize")

    @abstractmethod
    def predict(self, feed):
        """
        Run inference on input data.

        Must be implemented by subclasses to perform model prediction.

        Args:
            feed: Input data for inference.

        Returns:
            Model predictions.
        """
        raise NotImplementedError("DLRMBackend:predict")
