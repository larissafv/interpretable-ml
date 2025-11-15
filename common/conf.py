"""Project-wide configuration constants.

This module centralizes environment-driven paths used across the project:

- MODEL_PATH: Filesystem path to the serialized ML model (Keras/TensorFlow).
- PLOT_PATH: Directory where generated plot artifacts will be written.

Both can be overridden with environment variables of the same name.
"""

import os
from pathlib import Path
from typing import Final

MODEL_PATH: Final = Path(
    os.environ.get("MODEL_PATH")
)
PLOT_PATH: Final = Path(
    os.environ.get("PLOT_PATH")
)
