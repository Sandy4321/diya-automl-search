from abc import ABC, abstractmethod
import os
import json
from utils.logger import Logger


class Base(ABC):
    def __init__(self, name, args):
        self.logger = Logger(name, verbose=True, args=args)
        path = os.path.join(self.logger.log_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(vars(args), f)

    @abstractmethod
    def search(self):
        pass
