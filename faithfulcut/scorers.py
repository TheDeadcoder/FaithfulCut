import numpy as np

from faithfulcut.replay import History


def qtf(history: History, t: int) -> np.ndarray:
    return history.last_duration
