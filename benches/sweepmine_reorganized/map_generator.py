import random
from .core_generator import core_size


def _map_generator(dif:int , seed: int) -> list[list[int]]:
    size = core_size(dif) + (2 if dif % 2 else 4)
    
    """Sample active positions on a 3x3 core, excluding its center."""
    available=size*size-1
    active_count = available // 8
    if type(active_count) is not int or not 1 <= active_count <= available:
        raise ValueError("active_count must be between 1 and the available plots")
    positions = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) != (size//2, size//2)
    ]
    selected = random.Random(seed).sample(positions, active_count)
    map = [[0 for _ in range(size)] for _ in range(size)]
    for row, col in selected:
        map[row][col] = 1
    return map

#print(_map_generator(1,3,5))
