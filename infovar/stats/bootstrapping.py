from typing import Tuple, Optional

import numpy as np

def degeneracy_handling(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    duplicates_y: bool=False,
    duplicates_x: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Additive gaussian noise to avoid degeneracy.
    In practice, `duplicates_y` must always be True while `duplicates_x` is only True when `X` is a bootstrap sample.
    """
    if X is None and duplicates_y:
        Y += np.random.normal(0, 1e-3, Y.shape)
    if X is not None and duplicates_x:
        X += np.random.normal(0, 5e-2, X.shape)
    return Y, X if X is not None else None
