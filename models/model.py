# Model Abstract Base Class with a consistent interface

from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def forward_pass(inputs: list) -> list:
        """Runs a forward pass on the model instance, takes in a list of inputs and returns a list of outputs at corresponding indices"""
        raise NotImplementedError("Please implement Model Forward Pass functionality")

