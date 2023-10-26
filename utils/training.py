import numpy as np
import pickle as pkl
import time
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
print(script_dir)