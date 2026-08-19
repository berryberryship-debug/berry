import math
class SolitonEngine58:
    def __init__(self, amplitude: float = 1.0, velocity: float = 1.0, damping: float = 0.1):
        self.amplitude = amplitude
        self.velocity = velocity
        self.damping = damping

    def compute_wave_packet(self, x: float, t: float) -> float:
        try:
            arg = x - (self.velocity * t)
            return float(self.amplitude) * (1.0 / math.cosh(arg)) * math.exp(-self.damping * abs(arg))
        except OverflowError:
            return 0.0
