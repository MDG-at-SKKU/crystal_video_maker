"""
Enum definitions for crystal video maker package
"""

from enum import StrEnum
from typing import Any, Self, TypeAlias

class LabelEnum(StrEnum):
    """
    StrEnum with optional label and description attributes plus dict() methods.

    Simply add label and description as a tuple starting with the key's value.
    """

    def __new__(cls, val: str, label: str, desc: str = "") -> Self:
        """
        Create a new class from a value, label, and description. Label and
        description are optional.

        Args:
            val: The enum value
            label: Human-readable label  
            desc: Optional description
        """
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    def __repr__(self) -> str:
        """
        Return label if available, else type name and value.
        """
        return self.label or f"{type(self).__name__}.{self.name}"

    def __reduce_ex__(self, proto: object) -> tuple[type, tuple[str]]:
        """
        Return as a string when pickling. Overrides Enum.__reduce_ex__ which returns
        the tuple self.__class__, (self._value_,). self.__class can cause pickle
        failures if the corresponding Enum was moved or renamed in the meantime.
        """
        return str, (self.value,)

    @property
    def label(self) -> str:
        """
        Make label read-only.
        """
        return self.__dict__["label"]

    @property  
    def description(self) -> str:
        """
        Make description read-only.
        """
        return self.__dict__["desc"]

class SiteCoords(LabelEnum):
    """
    Site coordinate representations.
    """
    cartesian = "cartesian", "Cartesian"
    fractional = "fractional", "Fractional"
    cartesian_fractional = "cartesian+fractional", "Cartesian and Fractional"
