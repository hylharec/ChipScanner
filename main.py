import numpy as np
import cv2
from pycromanager import Core
from time import sleep

import ChipScanner
import CorvusDriver

# Using PylabSAS
#from PyLabSAS.instruments.motion_controller.smc_corvus import SMCCorvusXYZ

#scanner = ChipScanner.ChipScanner(2899, 2899, 50, 50, 50, [640, 512], 15)
scanner = ChipScanner.ChipScanner(3856, 3643, 60, 60, 50, [640, 512], 15)
scanner.scan(40, 30, 30)
"""scanner.xyz_stage.move_xyz_abs(0, 0, 0)
scanner.find_best_focus(
    z_step_size_um=4,
    max_tries=40,
    min_step_size=0.25
)"""

"""corvus = CorvusDriver.SMCCorvusXYZ(
    port="COM9",
    baud_rate=57600,
    bytesize="EIGHTBITS",
    stopbits="STOPBITS_ONE",
    parity="PARITY_NONE",
    timeout=1,
    accel=1,
    velocity=1
)

corvus.move_xyz_abs(0.0, 0.0, 0.0) # go to zero

positions = [
    (-1, 0, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (0, 0, 0)
]

for (x, y, z) in positions:
    corvus.move_xyz_abs(x, y, z)
    sleep(1.0)

input("End of program, enter to exit...")"""

