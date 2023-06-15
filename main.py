import ChipScanner

#scanner = ChipScanner.ChipScanner(2899, 2899, 50, 50, 50, [640, 512], 15)


scanner = ChipScanner.ChipScanner(
    x_end_um=3856,
    y_end_um=3643,
    margins_um=(100, 100),
    coverage_overlap=0.3,
    lens=5,
    camera_res=[640, 512],
    pixel_pitch_um=15.0,
    cam2motor_angle=0.020
)
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

