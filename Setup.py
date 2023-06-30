import queue
import threading
import cv2
import numpy as np
from pycromanager import Core
import logging

import CorvusDriver

class Setup:
    def __init__(self, logger: logging.Logger):
        # ########################################################################
        self.logger = logger
        # Initialize connection with MM core server
        self._core = None

        # Queue used to ask thread to stop
        self._control_queue = queue.Queue(maxsize=32)
        self._cam_view_thread = threading.Thread(target=self._cam_view_thread_update, args=(self._control_queue,))

    def setup(self):
        """
        Start the experiment setup sequence:
        - Set exposure used throughout experiment
        - Set zeros (top left hand corner) of chip area
        - Set end position (bottom right hand corner) of chip area

        Returns:
            - (x_end_um, y_end_um): (int, int) Tuple of designated end position of chip area with regards to set zero position.
        """

        # ########################################################################
        self.logger.info("Connecting to MicroManager core server...")
        try:
            self._core: Core = Core()
        except Exception:
            self.logger.error("Could not connect to MicroManager core server, make sure it is running. Exit.\n\n")
            exit()

        # ########################################################################
        # Initialize connection with xyz controller
        self.logger.info("Connecting to XYZ instrument...")
        xyz_stage = CorvusDriver.SMCCorvusXYZ( # unit is um
            port="COM9",
            baud_rate=57600,
            bytesize="EIGHTBITS",
            stopbits="STOPBITS_ONE",
            parity="PARITY_NONE",
            timeout=0.1,
            accel=5000,
            velocity=15000
        )

        self.start()

        # Disable auto-exposure on a spot of the chip were exposure is acceptable
        self._core.set_property("Raptor Ninox Camera 640", "Exposure: Auto", "On")
        xyz_stage.set_joystick(True)
        self.logger.info("Setting exposure: Please use the joystick to go to an area of the chip were the autoexposure will be set constant.")
        input("\nThen press enter to continue...")
        self._core.set_property("Raptor Ninox Camera 640", "Exposure: Auto", "Off")

        # Set zero of XY stage on first corner of chip
        self.logger.info("Go to top left hand corner of chip with regards to the image viewed with the program.")
        self.logger.info("(Change z to set a good starting focus)")
        self.logger.info("Cross needs to be INSIDE chip corner or autofocus might take edges into account.")
        input("\nThen press enter to continue...")

        xyz_stage.set_zero()

        # Define X_END and Y_END by selecting last corner of chip
        self.logger.info("Go to bottom right hand corner of chip with regards to the image viewed with the program.")
        self.logger.info("(Changing z is irrelevent for this step)")
        self.logger.info("Cross needs to be INSIDE chip corner or autofocus might take edges into account.")
        input("\nThen press enter to continue...")

        pos = xyz_stage.get_pos(return_list=True)
        x_end_um, y_end_um = int(pos[0]), int(pos[1])

        # End setup sequence to get ready for subsequent scan
        xyz_stage.close_instrument()

        self.stop()
        self._core = None

        return (x_end_um, y_end_um)

    def start(self):
        """
        Starts the thread if has not been done already.
        """
        if not self._cam_view_thread.is_alive():
            self._cam_view_thread.start()

    def stop(self):
        """
        Stops update thread if is running. Blocking until thread is closed.
        """
        if self._cam_view_thread.is_alive():
            self._control_queue.put("Exit")
            self._cam_view_thread.join()

    def _cam_view_thread_update(self, control_queue: queue.Queue):

        cv2.namedWindow("cv_win", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        while True:
            if self._core is not None:
                self._core.snap_image()

                # Try might fail on the first line if MM fails to answer correctly (just ignore such cases and retry)
                try:
                    tagged_image = self._core.get_tagged_image()

                    # Images from camera should normally be 640x512 14bit deep grayscale.
                    pixels = np.reshape(
                        tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 1]
                    ).astype(np.uint16)

                    # Reshape to (_, _) in case shape was (_, _, 1)
                    pixels = pixels.reshape((pixels.shape[0], pixels.shape[1]))

                    # Apply values scalar multiplication to convert from 14bit to 16bit range
                    pixels = pixels << 2

                    # Draw cross in the middle of the image
                    H, W = pixels.shape[0], pixels.shape[1]
                    pixels = cv2.line(pixels, (W//2 - 30, H//2), (W//2 + 30, H//2), (255, 0, 0), 5)
                    pixels = cv2.line(pixels, (W//2, H//2 - 30), (W//2, H//2 + 30), (255, 0, 0), 5)

                    cv2.imshow("cv_win", pixels)
                    cv2.waitKey(50)

                except Exception:
                    self.logger.warning("Error while getting image from MM. Ignoring...")


            # Handle thread stop command
            if not control_queue.empty() and control_queue.get() == "Exit":
                break

        cv2.destroyWindow("cv_win")

