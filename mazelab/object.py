from dataclasses import dataclass
from dataclasses import field

@dataclass
class Object:
    r""" defines an object with some of its properties.
    An object can be switches or doors or obstacles """
    name: str
    value: int
    rgb: tuple
    colorname: str
    impassable: bool
    positions: list = field(default_factory=list)
