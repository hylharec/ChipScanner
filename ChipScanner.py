import numpy as np
from pycromanager import Core
import cv2
from time import sleep

import CorvusDriver
import ImageMerger

class ChipScanner:
    def __init__(
        self,
        x_end_um,
        y_end_um,
        margins_um: tuple = (0, 0),
        coverage_overlap: float = 0.3,
        lens: int = 5,
        camera_res: list = [640, 512], # [W, H]
        pixel_pitch_um: float = 15.0,
        cam2motor_angle: float = 0.020
    ):
        """
        x_end_um/y_end_um: (int) coordinates where the end corner of chip is in the middle of the camera
        margins_um: (tuple of int) margins around the chip to include correcly it's edges in the final picture
        coverage_overlap: (float) percentage [0.0; 1.0[ of overlap between each individual picture (does not take elliptic mask into account)
        lens: (int) microscope magnification (x5, x20 or x50)
        camera_res: ([int, int]) resolution of picture snapped through MicroManager
        pixel_pitch: (float) pixel size in um (get from camera datasheet)
        cam2motor_angle: (float) rotational misalignment between camera and XY stage (in rad)
        """
        # To match with microscope physical parameters
        self.camera_res = list(camera_res)
        self.pixel_pitch = pixel_pitch_um # um
        self.lens = lens # magnification
        self.IMG_UM_WIDTH = pixel_pitch_um * camera_res[0] / float(lens)
        self.IMG_UM_HEIGHT = pixel_pitch_um * camera_res[1] / float(lens)
        self.PXL_PER_UM = float(lens) / pixel_pitch_um

        self.cam2motor_angle = cam2motor_angle

        # Determines the number of pictures to take depending on the area of the chip to cover
        self.X_END = x_end_um
        self.Y_END = y_end_um
        self.coverage_overlap = coverage_overlap
        self.margins_um = margins_um
        margin_x_um, margin_y_um = margins_um

        self.X_NUM = (self.X_END + margin_x_um * 2.0) / (self.IMG_UM_WIDTH * (1.0 - coverage_overlap)) + 1
        self.Y_NUM = (self.Y_END + margin_y_um * 2.0) / (self.IMG_UM_HEIGHT * (1.0 - coverage_overlap)) + 1

        # Generate scan (x, y) tuples
        self.positions = self._snakescan(
            -margin_x_um,
            -margin_y_um,
            self.X_END + margin_x_um,
            self.Y_END + margin_y_um,
            self.X_NUM,
            self.Y_NUM
        )

        # ########################################################################
        # Initialize connection with MM core server
        print("Connecting to MicroManager core server...")
        self.core = Core()
        self.core.set_property("Raptor Ninox Camera 640", "Exposure: Auto", "Off")

        # Set image grayscale format
        self._pixel_shape = 1 # 4 -> RGBA, 1 -> grayscale
        self._INPUT_BIT_DEPTH = 16384
        self.BIT_DEPTH = None
        self._image_dtype = None
        if np.log2(self._INPUT_BIT_DEPTH) > 32:
            self._image_dtype = np.uint64
            self.BIT_DEPTH = np.power(2, 64)
        elif np.log2(self._INPUT_BIT_DEPTH) > 16:
            self._image_dtype = np.uint32
            self.BIT_DEPTH = np.power(2, 32)
        elif np.log2(self._INPUT_BIT_DEPTH) > 8:
            self._image_dtype = np.uint16
            self.BIT_DEPTH = np.power(2, 16)
        else:
            self._image_dtype = np.uint8
            self.BIT_DEPTH = np.power(2, 8)
        # Following attribute is used in picture snap thread function
        self._INPUT_TO_OUTPUT_BIT_DEPTH_MULT = int(np.power(2.0, (np.log2(self.BIT_DEPTH) - np.log2(self._INPUT_BIT_DEPTH))))


        # ########################################################################
        # Config a bunch of camera parameters
        print("Configuring camera...")
        #do_mirror_x = int(self.core.get_property('OpenCVgrabber', 'Flip X'))
        #self.core.set_property('OpenCVgrabber', 'Flip X', 1 - do_mirror_x)

        # ########################################################################
        # Initialize connection with xyz controller
        print("Connecting to XYZ instrument...")
        self.xyz_stage = CorvusDriver.SMCCorvusXYZ( # unit is um
            port="COM9",
            baud_rate=57600,
            bytesize="EIGHTBITS",
            stopbits="STOPBITS_ONE",
            parity="PARITY_NONE",
            timeout=0.1,
            accel=5000,
            velocity=15000
        )
        self.xyz_stage.set_joystick(False)

    def __del__(self):
        # Re-enable joystick before quitting
        self.xyz_stage.set_joystick(True)

    def _snakescan(self, xi, yi, xf, yf, xn, yn):
        """
        Scan pixels in a snake pattern along the x-coordinate then y-coordinate.
        Skips one in two positions to create a pattern like:
        +.+.+.+.+.+.+
        .+.+.+.+.+.+.
        +.+.+.+.+.+.+
        For better filling of space with rounded pictures.
        """
        x_list = np.linspace(xi, xf, xn)
        y_list = np.linspace(yi, yf, yn)

        positions = []
        index = 0
        for i in range(len(x_list)):
            for j in range(len(y_list)):
                if i % 2 == 0:
                    if index % 2 == 0:
                        positions.append((x_list[i], y_list[j]))
                else:
                    if index % 2 == 0:
                        positions.append((x_list[i], y_list[len(y_list) - j - 1]))
                index += 1
        return positions

    def scan(self, ellipse_offset_x: int = 300, ellipse_offset_y: int = 300, blur_strength: int = 50):
        # ########################################################################
        # Start exploring the chip and taking pictures
        print("Scanning area...")
        stop = 0
        for i in range(len(self.positions)):
            (x, y) = np.round(self.positions[i], 3)

            # Printing scanning progress
            progress = int(i * 50.0 / (len(self.positions) - 1)) # progress goes from 0 to 50
            print(f"({round(x, 2)}, {round(y, 2)}) : \n" + progress * "#" + (50 - progress) * "-")

            # Go to position (x, y), in um
            self.xyz_stage.move_xyz_abs(x, y, 0)
            sleep(0.2)

            focus_z = self.find_best_focus(2.0, 20, 0.25)
            print(f"Focus z = {focus_z}")

            # Snap a picture
            pixels = self._get_raw_camera_image()

            # Get real position
            pos = np.round(self.xyz_stage.get_pos(True), 3)
            #print(f"({x_m}, {y_m}) => ({pos[0]}, {pos[1]})")
            x, y = pos[0], pos[1]

            # ================= WORKS ? Was after imwrite
            # Apply values scalar multiplication in case bit depth is not a multiple of 2
            pixels = pixels * self._INPUT_TO_OUTPUT_BIT_DEPTH_MULT

            #cv2.imwrite(f"img_to_merge/{i}_{len(self.positions)}_{x}_{y}.png", cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(f"img_to_merge/{i}_{len(self.positions)}_{x}_{y}.png", pixels)

            #stop += 1
            if stop >= 10:
                break

        # ########################################################################
        # Once all pictures were taken, they need to be blended together to form one big image

        print("Merging images...")
        merger = ImageMerger.ImageMerger(
            img_path_base="img_to_merge",
            pxl_per_um=self.PXL_PER_UM,
            lens=self.lens,
            #merge_img_size=[int(self.camera_res[0] + self.PXL_PER_UM * self.X_END), int(self.camera_res[1] + self.PXL_PER_UM * self.Y_END)],
            merge_img_size=[
                int(self.Y_NUM * self.camera_res[0]),
                int(self.X_NUM * self.camera_res[1]),
            ],
            margins_um=[self.margins_um[0], self.margins_um[1]],
            x_end_um=self.X_END,
            y_end_um=self.Y_END,
            cam2motor_angle=self.cam2motor_angle,
        )
        merger.load()
        merger.merge(
            offset=[ellipse_offset_x, ellipse_offset_y],
            blur_level=blur_strength,
            clean_after=False
        )

        print("Done.")

    def find_best_focus(self, z_step_size_um: float, max_tries: int, min_step_size: float):
        """
        This method tries to find the Z position that provides the best focus. It also returns the best Z value found.
        It needs to be called after moving to the desired XY position and right before taking the picture.
        Method ends before max number of tries if step size became small enough (converged).
        """
        last_score = None
        best_z = None
        nb_tries = 0

        direction = "down"
        already_inverted_direction_once = False

        pos = []
        while len(pos) == 0:
            pos = np.abs(np.round(self.xyz_stage.get_pos(True), 3))
        x, y, current_z = pos[0], pos[1], pos[2]

        while nb_tries < max_tries:
            nb_tries += 1

            score = self._get_image_focus_score(self._get_raw_camera_image(), current_z)
            #print(f"Z: {current_z}, Score: {score}, dir: {direction}, step: {z_step_size_um}")

            if last_score is None:
                best_z = current_z
            else:
                if score >= last_score: # Stay in same direction to keep improving
                    best_z = current_z
                else: # Score got worse, invert direction or stop algorithm
                    # Invert direction
                    if direction == "up":
                        direction = "down"
                    else:
                        direction = "up"
                    if already_inverted_direction_once:
                        # Go in oposite direction with smaller step size

                        # If step size has become too small, solution considered to have been found
                        if z_step_size_um <= min_step_size:
                            self.xyz_stage.move_xyz_abs(x, y, best_z)
                            return best_z
                        else:
                            z_step_size_um /= 2
                    else:
                        # Do not reduce step size for first inversion because initial direction might have been wrong.
                        already_inverted_direction_once = True

            # Compute next z pos to explore
            if direction == "up":
                current_z -= z_step_size_um
            else:
                current_z += z_step_size_um

            if current_z < -400:
                current_z = -400
            elif current_z > 400:
                current_z = 400

            # Go to next z
            self.xyz_stage.move_xyz_abs(x, y, current_z)

            last_score = score

        # Maximum number of tries reached, go to best solution and return best z found.
        self.xyz_stage.move_xyz_abs(x, y, best_z)
        return best_z

    def _get_image_focus_score(self, image, z):
        """
        Method used exclusiely by the autofocus method to rate an image focus.
        Input image is the same format as returned by self._get_raw_camera_image().
        """
        # Image snapped from photoemission camera is grayscale 14bit
        # Apply Gaussian blur (params taken from Opencv Autofocus code)
        image = cv2.GaussianBlur(image, (7, 7), sigmaX=1.5, sigmaY=1.5)
        # Find edges (params taken from Opencv Autofocus code)

        image = image >> 8
        image = np.uint8(image)
        image = cv2.equalizeHist(image)

        #cv2.imwrite(f"focus/image_{z}.png", image)

        edges: np.array = cv2.Canny(image, threshold1=0, threshold2=30, apertureSize=3)

        #cv2.imshow(f"focus/edge_{z}.png", edges)
        #cv2.waitKey(10)

        # Score is defined as average value of edge image
        return edges.mean()

    def _get_raw_camera_image(self):
        self.core.snap_image()

        tagged_image = self.core.get_tagged_image()
        (W, H) = (tagged_image.tags["Width"], tagged_image.tags["Height"])

        pixels = np.reshape(
            tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 1]
        ).astype(self._image_dtype)

        return pixels
