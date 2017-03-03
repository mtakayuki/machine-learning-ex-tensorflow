import os
from datetime import datetime


def subdir(base='logs'):
    return os.path.join(base, datetime.now().strftime('%Y%m%d-%H%M%S'))
