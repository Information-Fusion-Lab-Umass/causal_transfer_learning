from dataclasses import dataclass

@dataclass
class Color:
    # door = (0, 255, 0)
    free = (255, 255, 255)
    obstacle = (0, 0, 0)
    agent = (255, 255, 0)
    switch = (47,79,79)
    prize = (165, 42, 42)
