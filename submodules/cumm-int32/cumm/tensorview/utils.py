def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

def align_up(a: int, b: int) -> int:
    return div_up(a, b) * b
