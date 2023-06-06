import numpy as np
import cv2
from pycromanager import Core
from time import sleep

import ChipScanner

# Using PylabSAS
#from PyLabSAS.instruments.motion_controller.smc_corvus import SMCCorvusXYZ

scanner = ChipScanner.ChipScanner(6400*3-1, 4800*3-1, 10, 10, 0.3, [640, 480], 1.0)
scanner.scan(300, 200, 80)
