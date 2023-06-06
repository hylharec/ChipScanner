import os
import glob
import numpy as np
import cv2
from pycromanager import Core
from time import sleep
from PIL import Image, ImageDraw, ImageChops, ImageFilter

# Using PylabSAS
#from PyLabSAS.instruments.motion_controller.smc_corvus import SMCCorvusXYZ

class ChipScanner:
    def __init__(self, x_end, y_end, x_num, y_num, lens: int = 5, camera_res: list = [640, 512], pixel_pitch: float = 15.0):
        """
        x_end/y_end: (int) in um
        x_num/y_num: (int) number of pictures to get on each axis
        lens: (int) microscope magnification (x5, x20 or x50)
        camera_res: ([int, int]) resolution of picture snapped through MicroManager
        pixel_pitch: (float) pixel size in um (get from camera datasheet)
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

        # ########################################################################
        # Initialize connection with MM core server
        print("Connecting to MicroManager core server...")
        self.core = Core()

        # ########################################################################
        # Config a bunch of camera parameters
        print("Configuring camera...")
        do_mirror_x = int(self.core.get_property('OpenCVgrabber', 'Flip X'))
        self.core.set_property('OpenCVgrabber', 'Flip X', 1 - do_mirror_x)

        # ########################################################################
        # Initialize connection with xyz controller
        print("Connecting to XYZ instrument...")
        #xyz_stage = SMCCorvusXYZ()
        # Set current coordinates to (0, 0, 0)
        #xyz_stage.set_zero()

    def _snakescan(self, xi, yi, xf, yf, xn, yn):
        """
        Scan pixels in a snake pattern along the x-coordinate then y-coordinate
        """
        x_list = np.linspace(xi, xf, xn)
        y_list = np.linspace(yi, yf, yn)

        positions = []
        for x in x_list:
            for i in range(len(y_list)):
                if x % 2 == 0:
                    positions.append((x, y_list[i]))
                else:
                    positions.append((x, y_list[len(y_list) - i - 1]))
        return positions

    def scan(self, ellipse_offset_x: int = 300, ellipse_offset_y: int = 300, blur_strength: int = 50):
        # ########################################################################
        # Start exploring the chip and taking pictures
        print("Scanning area...")
        (W, H) = (0, 0)
        for i in range(len(self.positions)):
            (x, y) = self.positions[i]
            progress = int(i * 50.0 / (len(self.positions) - 1)) # progress goes from 0 to 50
            print(f"({round(x, 2)}, {round(y, 2)}) : \n" + progress * "#" + (50 - progress) * "-")
            # Go to position (x, y), in um
            #xyz_stage.move_xyz_abs(x, y, 0)

            # wait until position is reached
            #while abs(xyz_stage.get_pos_x() - x) > 0.1 or abs(xyz_stage.get_pos_y() - y) > 0.1:
            #    sleep(20)

            # Snap a picture
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            (W, H) = (tagged_image.tags["Width"], tagged_image.tags["Height"])
            pixels = np.reshape(
                tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"], 4]
            )
            cv2.imwrite(f"img_to_merge/{x}_{y}.jpg", pixels)

        # ########################################################################
        # Once all pictures were taken, they need to be blended together to form one big image

        print("Merging images...")
        self._merge_images(
            "img_to_merge",
            self.PXL_PER_UM,
            [int(self.camera_res[0] + self.PXL_PER_UM * self.X_END), int(self.camera_res[1] + self.PXL_PER_UM * self.Y_END)],
            [ellipse_offset_x, ellipse_offset_y],
            blur_strength,
            clean_after=True,
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
            background = Image.new('RGB', size=(merge_img_size[0], merge_img_size[1]),
                                        color=(255, 255, 255))

            # Ignore these filenames when loading images.
            ignore_filenames = ["result.jpg"]

            # Dictionary to save loaded images.
            images = {}

            # Load images and put them into images dict().
            for filename in glob.glob(os.path.join(img_path_base, '*.jpg')):
                basename = os.path.basename(filename)
                if basename not in ignore_filenames:
                    coords = basename.strip(".jpg")  # fname without extension
                    x = float(coords.split('_')[0])  # X coordinate
                    y = float(coords.split('_')[1])  # Y coordinate
                    # images[x, y] = Image.open(filename)  # same image in dict()
                    images[x, y] = Image.open(filename).convert('RGBA')

            # Merge overlaping images.
            for coords, image in images.items():
                x, y = coords  # coords are micrometers
                x_px = int(x * pxl_per_um)
                y_px = int(y * pxl_per_um)
                image = self._crop_to_circle(image, blur_level=blur_level, offset=offset)

                background.paste(image, (x_px, y_px), image)

            # Save the resulting image.
            path_result = os.path.join(img_path_base, 'result.jpg')

            # Remove result if exists
            try:
                os.remove(path_result)
            except:
                pass

            background.save(path_result, "JPEG")

            # Clean individual images after result was saved if specified
            if clean_after:
                for filename in glob.glob(os.path.join(img_path_base, '*.jpg')):
                    basename = os.path.basename(filename)
                    if basename not in ignore_filenames:
                        os.remove(filename)

    def _crop_to_circle(self, im: Image, blur_level: float = 0, offset: list = [500, 500]):
        """
        Crop the image into a blured circle.
        """

        bigsize = (im.size[0] * 3, im.size[1] * 3)
        bigsize2 = (im.size[0] * 3 - offset[0], im.size[1] * 3 - offset[1])

        mask = Image.new('L', bigsize, 0)
        ImageDraw.Draw(mask).ellipse((offset[0], offset[1]) + bigsize2, fill=255)
        mask = mask.resize(im.size, Image.ANTIALIAS)
        mask = ImageChops.darker(mask, im.split()[-1])
        mask = mask.filter(ImageFilter.GaussianBlur(blur_level))
        im.putalpha(mask)

        return im



