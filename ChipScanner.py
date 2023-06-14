import os
import glob
import numpy as np
import cv2
from pycromanager import Core
from time import sleep
from PIL import Image

# Using PylabSAS
#from PyLabSAS.instruments.motion_controller.smc_corvus import SMCCorvusXYZ

import CorvusDriver

class ChipScanner:
    def __init__(self, x_end, y_end, x_num, y_num, lens: int = 5, camera_res: list = [640, 512], pixel_pitch: float = 15.0, cam2motor_angle: float = 0.020):
        """
        x_end/y_end: (int) in um
        x_num/y_num: (int) number of pictures to get on each axis
        lens: (int) microscope magnification (x5, x20 or x50)
        camera_res: ([int, int]) resolution of picture snapped through MicroManager
        pixel_pitch: (float) pixel size in um (get from camera datasheet)
        cam2motor_angle: (float) rotational misalignment between camera and XY stage (in rad)
        """
        self.X_END = x_end # um
        self.Y_END = y_end # um
        self.X_NUM = x_num
        self.Y_NUM = y_num
        # Generate scan (x, y) tuples
        self.positions = self._snakescan(0, 0, self.X_END, self.Y_END, self.X_NUM, self.Y_NUM)

        # To match with microscope physical parameters
        self.camera_res = list(camera_res)
        self.pixel_pitch = float(pixel_pitch) # um
        self.lens = lens # magnification
        self.IMG_UM_WIDTH = pixel_pitch * camera_res[0] / float(lens)
        self.IMG_UM_HEIGHT = pixel_pitch * camera_res[1] / float(lens)
        self.PXL_PER_UM = float(lens) / float(pixel_pitch)

        self.cam2motor_angle = cam2motor_angle

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
        # Set current coordinates to (0, 0, 0)
        #self.xyz_stage.set_zero()

    def _snakescan(self, xi, yi, xf, yf, xn, yn):
        """
        Scan pixels in a snake pattern along the x-coordinate then y-coordinate
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
        (W, H) = (0, 0)
        stop = 0
        for i in range(len(self.positions)):
            (x, y) = np.round(self.positions[i], 3)
            progress = int(i * 50.0 / (len(self.positions) - 1)) # progress goes from 0 to 50
            print(f"({round(x, 2)}, {round(y, 2)}) : \n" + progress * "#" + (50 - progress) * "-")
            # Go to position (x, y), in um

            # Convert coords from cam space to motors space
            x_m, y_m = self.coords_cam2motor(x, y)

            self.xyz_stage.move_xyz_abs(x_m, y_m, 0)
            sleep(0.2)

            focus_z = self.find_best_focus(2.0, 20, 0.25)
            print(f"Focus z = {focus_z}")

            # wait until position is reached => Should already be handled by previous function call
            #while abs(self.xyz_stage.get_pos_x() - x) > 0.1 or abs(self.xyz_stage.get_pos_y() - y) > 0.1:
            #    sleep(0.100)

            # Snap a picture
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            (W, H) = (tagged_image.tags["Width"], tagged_image.tags["Height"])
            pixels = np.reshape(
                tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 1]
            ).astype(self._image_dtype)

            # Get real position
            pos = np.abs(np.round(self.xyz_stage.get_pos(True), 3))
            #print(f"({x_m}, {y_m}) => ({pos[0]}, {pos[1]})")
            x, y = np.round(self.coords_motor2cam(pos[0], pos[1]), 3)

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
        self._merge_images(
            "img_to_merge",
            self.PXL_PER_UM,
            [int(self.camera_res[0] + self.PXL_PER_UM * self.X_END), int(self.camera_res[1] + self.PXL_PER_UM * self.Y_END)],
            [ellipse_offset_x, ellipse_offset_y],
            blur_strength,
            clean_after=False,
        )

        print("Done.")

    def _merge_images(
        self,
        img_path_base = "img_to_merge",
        pxl_per_um: float = 640,
        merge_img_size = [1500, 1500],
        offset=[640//2, 512//2],
        blur_level=50,
        clean_after: bool = True,
    ):
            # Blank image used as background.
            np_background = np.zeros((merge_img_size[1]+500, merge_img_size[0]+500), dtype=self._image_dtype)
            bg_weights = np.zeros((merge_img_size[1]+500, merge_img_size[0]+500), dtype=self._image_dtype)
            weight = np.zeros((self.camera_res[1], self.camera_res[0]), dtype=self._image_dtype)

            offset_x, offset_y = offset
            cv2.ellipse(weight, (self.camera_res[0]//2, self.camera_res[1]//2), (self.camera_res[0]//2-offset_x, self.camera_res[1]//2-offset_y), 0, 0, 360, (255), -1)
            weight = cv2.GaussianBlur(weight, (99, 99), sigmaX=blur_level, sigmaY=blur_level)

            # Ignore these filenames when loading images.
            ignore_filenames = ["result.png", "result_np.png", "result1.png", "result2.png", "result3.png", "result4.png", ]

            # Dictionary to save loaded images.
            images = []

            # Load images and put them into images dict().
            for filename in glob.glob(os.path.join(img_path_base, '*.png')):
                basename = os.path.basename(filename)
                if basename not in ignore_filenames:
                    coords = basename.strip(".png")  # filename without extension
                    i = int(coords.split('_')[0]) # order in wich img was taken
                    x = self.X_END - float(coords.split('_')[2])  # X coordinate
                    y = float(coords.split('_')[3])  # Y coordinate

                    #image = Image.open(filename).convert('RGBA')
                    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                    if i >= len(images):
                        images.append((x, y, image))
                    else:
                        images.insert(i, (x, y, image))

            # Load vignette effect correction lookup tables from textfile
            # depending on current lens.
            # Load LUT that correspond to current lens
            try:
                vignette_corr_lut = np.loadtxt(f"x{self.lens}_vignette_lut.txt")
            except Exception as err:
                print(f"Error: could not load vignette correction LUT: {err}")
                exit(-1)
            vignette_corr_lut = vignette_corr_lut.reshape((self.camera_res[0], self.camera_res[1]))

            H, W = self.camera_res[1], self.camera_res[0]

            # Merge overlaping images.
            index = 0
            for x, y, image in images:
                print(f"Merging img ({x}, {y})...")
                # coords are micrometers
                x_px = int(np.round(x * pxl_per_um))
                y_px = int(np.round(y * pxl_per_um))

                # Apply vignette effect correction
                image = image.astype(np.float128)
                image[1:H-1, 1:W-1] = np.multiply(image[1:H-1, 1:W-1], vignette_corr_lut)
                image = image.astype(self._image_dtype)

                #gray_image,_,_,_ = cv2.split(np.array(image))

                np_background[y_px+1:y_px+1+H-2, x_px+1:x_px+1+W-2] += np.multiply(
                    #gray_image[1:H-1, 1:W-1],
                    image[1:H-1, 1:W-1],
                    weight[1:H-1, 1:W-1]
                ) // 255
                bg_weights[y_px+1:y_px+1+H-2, x_px+1:x_px+1+W-2] += weight[1:H-1, 1:W-1]

                #index += 1
                if index > 100:
                    break

            # Cells == 0 => no pixel was added, weight can be set to 1.0 without consequence (to prevent division by zero on next step)
            bg_weights[bg_weights == 0] = 255

            np_background = np.divide(np_background * 255, bg_weights).astype(np.uint8)

            # Save the resulting image.
            path_result = os.path.join(img_path_base, 'result.png')

            # Remove result if exists
            try:
                os.remove(path_result)
            except:
                pass

            #np_background = cv2.equalizeHist(np_background)

            print(np_background.shape)
            print(np_background.dtype)
            cv2.imwrite(path_result, np_background)

            # Clean individual images after result was saved if specified
            if clean_after:
                for filename in glob.glob(os.path.join(img_path_base, '*.png')):
                    basename = os.path.basename(filename)
                    if basename not in ignore_filenames:
                        os.remove(filename)

    def coords_motor2cam(self, x, y):
        theta = self.cam2motor_angle
        x2 = np.cos(theta) * x + np.sin(theta) * y
        y2 = - np.sin(theta) * x + np.cos(theta) * y
        return x2, y2

    def coords_cam2motor(self, x, y):
        theta = self.cam2motor_angle
        x2 = np.cos(theta) * x - np.sin(theta) * y
        y2 = np.sin(theta) * x + np.cos(theta) * y
        return x2, y2

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
            print(f"Z: {current_z}, Score: {score}, dir: {direction}, step: {z_step_size_um}")

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

class ImageMerger:
    def __init__(
        self,
        img_path_base = "img_to_merge",
        pxl_per_um: float = 640,
        lens: int = 50,
        merge_img_size = [1500, 1500],
        ignored_filenames = ["result.png", "result_np.png", "result1.png", "result2.png", "result3.png", "result4.png"]
    ):
        self.img_path_base = img_path_base
        self.pxl_per_um = pxl_per_um
        self.lens = lens
        self.merge_img_size = merge_img_size
        self.ignored_filenames = ignored_filenames

        self.images = [] # contains tuples (x, y, image)
        self.images_dtype = None
        self.images_res = None

    def load(self):
        print("Load images to merge...")
        # Empty image list if it already contained something
        self.images = []
        self.images_dtype = None
        self.images_res = None
        # Load images and put them into images list.
        for filename in glob.glob(os.path.join(self.img_path_base, '*.png')):
            basename = os.path.basename(filename)
            if basename not in self.ignore_filenames:
                coords = basename.strip(".png")  # filename without extension
                i = int(coords.split('_')[0]) # order in wich img was taken
                x = self.X_END - float(coords.split('_')[2])  # X coordinate
                y = float(coords.split('_')[3])  # Y coordinate

                #image = Image.open(filename).convert('RGBA')
                image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                self.images_dtype = image.dtype
                H, W = image.shape
                self.images_res = [H, W]

                if i >= len(self.images):
                    self.images.append((x, y, image))
                else:
                    self.images.insert(i, (x, y, image))
        print("...Done.")

    def merge(
        self,
        offset=[640//2, 512//2],
        blur_level=50,
        clean_after: bool = True
    ):
        if self.images_dtype is None:
            print("Warning: Skipping image merging because no image was loaded.")
        else:
            print("Merge loaded images...")
            # Blank image used as background.
            np_background = np.zeros((self.merge_img_size[1]+500, self.merge_img_size[0]+500), dtype=self.images_dtype)
            bg_weights = np.zeros((self.merge_img_size[1]+500, self.merge_img_size[0]+500), dtype=self.images_dtype)
            weight = np.zeros((self.images_res[1], self.images_res[0]), dtype=self.images_dtype)

            offset_x, offset_y = offset
            cv2.ellipse(
                img=weight,
                center=(self.images_res[0]//2, self.images_res[1]//2),
                axis=(self.images_res[0]//2-offset_x, self.images_res[1]//2-offset_y),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255),
                thickness=-1
            )
            weight = cv2.GaussianBlur(weight, (99, 99), sigmaX=blur_level, sigmaY=blur_level)

            # Load vignette effect correction lookup tables from textfile
            # depending on current lens.
            # Load LUT that correspond to current lens
            try:
                vignette_corr_lut = np.loadtxt(f"x{self.lens}_vignette_lut.txt")
            except Exception as err:
                print(f"Error: could not load vignette correction LUT: {err}")
                exit(-1)
            vignette_corr_lut = vignette_corr_lut.reshape((self.images_res[0], self.images_res[1]))

            H, W = self.images_res[1], self.images_res[0]

            # Merge overlaping images.
            index = 0
            for x, y, image in self.images:
                print(f"Merging img ({x}, {y})...")
                # coords are micrometers
                x_px = int(np.round(x * self.pxl_per_um))
                y_px = int(np.round(y * self.pxl_per_um))

                # Apply vignette effect correction
                image = image.astype(np.float128)
                image[1:H-1, 1:W-1] = np.multiply(image[1:H-1, 1:W-1], vignette_corr_lut)
                image = image.astype(self.images_dtype)

                #gray_image,_,_,_ = cv2.split(np.array(image))

                np_background[y_px+1:y_px+1+H-2, x_px+1:x_px+1+W-2] += np.multiply(
                    #gray_image[1:H-1, 1:W-1],
                    image[1:H-1, 1:W-1],
                    weight[1:H-1, 1:W-1]
                ) // 255
                bg_weights[y_px+1:y_px+1+H-2, x_px+1:x_px+1+W-2] += weight[1:H-1, 1:W-1]

                #index += 1
                if index > 100:
                    break

            # Cells == 0 => no pixel was added, weight can be set to 1.0 without consequence (to prevent division by zero on next step)
            bg_weights[bg_weights == 0] = 255

            np_background = np.divide(np_background * 255, bg_weights).astype(np.uint8)

            # Save the resulting image.
            path_result = os.path.join(self.img_path_base, 'result.png')

            # Remove result if exists
            try:
                os.remove(path_result)
            except:
                pass

            #np_background = cv2.equalizeHist(np_background)

            print(np_background.shape)
            print(np_background.dtype)
            cv2.imwrite(path_result, np_background)

            # Clean individual images after result was saved if specified
            if clean_after:
                for filename in glob.glob(os.path.join(self.img_path_base, '*.png')):
                    basename = os.path.basename(filename)
                    if basename not in self.ignore_filenames:
                        os.remove(filename)

            print("...Done.")

