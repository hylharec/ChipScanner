import os
import glob
import yaml
import cv2
import numpy as np

class ImageMerger:
    """
    Class to merge a set of pictures taken from the surface of a chip with the ChipScanner class.
    """
    def __init__(
        self,
        img_path_base = "img_to_merge",
        camera_params_yaml_filename: str = "camera_parameters.yml",
        ignored_filenames = [
            "result.png",
            "result_np.png",
            "result1.png",
            "result2.png",
            "result3.png",
            "result4.png",
            "result_calibrated.png"
        ]
    ):
        self.img_path_base = img_path_base
        self.ignored_filenames = ignored_filenames

        self.images = [] # contains tuples (x, y, image)
        self.images_dtype = None
        self.images_res = None

        with open(camera_params_yaml_filename, mode="r", encoding="utf-8") as f:
            yaml_dump: dict = yaml.safe_load(f)

            exp_key = "Experiment"
            self.X_END =            int(yaml_dump[exp_key].get("x_end_um", 0))
            self.Y_END =            int(yaml_dump[exp_key].get("y_end_um", 0))
            self.margins_um =       int(yaml_dump[exp_key].get("x_margin_um", 0)), int(yaml_dump[exp_key].get("y_margin_um", 0))
            self.lens =             int(yaml_dump[exp_key].get("lens", 5))
            self.pixel_pitch_um =   float(yaml_dump[exp_key].get("pixel_pitch_um", 15.0))
            self.cam2motor_angle =  float(yaml_dump[exp_key].get("cam2motor_angle", 0.0))

        self.pxl_per_um = float(self.lens) / self.pixel_pitch_um

    def load(self):
        print("Load images to merge...")
        # Empty image list if it already contained something
        self.images = []
        self.images_dtype = None
        self.images_res = None
        # Load images and put them into images list.
        for filename in glob.glob(os.path.join(self.img_path_base, '*.png')):
            basename = os.path.basename(filename)
            if basename not in self.ignored_filenames:
                coords = basename.strip(".png")  # filename without extension
                i = int(coords.split('_')[0]) # order in wich img was taken

                raw_x = float(coords.split("_")[2]) + float(self.margins_um[0])
                raw_y = float(coords.split("_")[3]) + float(self.margins_um[1])

                # Convert coordinate system to try to compensate
                # rotation difference between camera and motor axis.
                x, y = self.coords_motor2cam(raw_x, raw_y)

                #image = Image.open(filename).convert('RGBA')
                image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                self.images_dtype = image.dtype
                H, W = image.shape
                self.images_res = [W, H]

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
            margin_x_um, margin_y_um = self.margins_um
            self.merge_img_size = [
                int((self.X_END + 2 * margin_x_um) * self.pxl_per_um + self.images_res[0] + 100), # + 100 to try to avoid edge
                int((self.Y_END + 2 * margin_y_um) * self.pxl_per_um + self.images_res[1] + 100), # cases because of motors precision.
            ]

            print("Merge loaded images...")
            full_img_W, full_img_H = self.merge_img_size[0], self.merge_img_size[1]
            img_W, img_H = self.images_res[0], self.images_res[1]

            margin_x_um, margin_y_um = self.margins_um
            margin_x_px = int(margin_x_um * self.pxl_per_um)
            margin_y_px = int(margin_y_um * self.pxl_per_um)

            # Blank image used as background.
            # the margins are added on top of the specified width and height to make sure all
            # x and y values are positive (to be able to paste images).
            full_image = np.zeros((full_img_H + margin_y_px , full_img_W + margin_x_px), dtype=np.uint64)
            weight_map = np.zeros((full_img_H + margin_y_px, full_img_W + margin_x_px), dtype=self.images_dtype)
            weight = np.zeros((full_img_H, full_img_W), dtype=self.images_dtype)

            # Create white ellipse in black weight image (will be used as a mask)
            offset_x, offset_y = offset
            cv2.ellipse(
                img=weight,
                center=(img_W // 2, img_H // 2),
                axes=(img_W // 2 - offset_x, img_H // 2 - offset_y),
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
            vignette_corr_lut = vignette_corr_lut.reshape((img_H-2, img_W-2))

            image_bit_depth = int(np.power(2.0, np.dtype(self.images_dtype).itemsize * 8))

            # Merge overlaping images.
            index = 0
            for x, y, image in self.images:
                print(f"Merging img ({x}, {y})...")
                # coords are micrometers
                x_px = int(np.round((x + margin_x_um) * self.pxl_per_um))
                y_px = int(np.round((y + margin_y_um) * self.pxl_per_um))

                # Apply vignette effect correction
                image = image.astype(np.uint64)
                image[1:img_H-1, 1:img_W-1] = np.multiply(
                    image[1:img_H-1, 1:img_W-1].astype(np.float64),
                    (vignette_corr_lut).astype(np.float64)
                ).astype(np.uint64)

                for row in image:
                    row[row > image_bit_depth - 1] = image_bit_depth - 1

                # To apply the elliptic mask on the current image, a higher dtype is casted to prevent
                # overflow during computation. The the masked image is recasted into the correct dtype.
                masked_image = np.multiply(
                    #gray_image[1:img_H-1, 1:img_W-1],
                    image[1:img_H-1, 1:img_W-1].astype(np.uint64),
                    weight[1:img_H-1, 1:img_W-1].astype(np.uint64)
                ) >> 8 # (// 255 because the weights are represented as uint8 (1.0 <=> 255))

                full_image[y_px+1:y_px+1+img_H-2, x_px+1:x_px+1+img_W-2] += masked_image
                weight_map[   y_px+1:y_px+1+img_H-2, x_px+1:x_px+1+img_W-2] += weight[1:img_H-1, 1:img_W-1]

                #index += 1
                if index > 100:
                    break

            # Cells == 0 => no pixel was added, weight can be set to 1.0 without consequence (to prevent division by zero on next step)
            weight_map[weight_map == 0] = 255

            # Once all images were merged, apply a per pixel intensity correction depending on where the sum of weights
            # of overlaping pixels does not add up to 1.
            # For exemple, if one pixel is on the overlap of 3 images with weights of respectively 0.1, 0.4 and 0.2,
            # the sum only adds up to 0.7. Thus for the averaging process to be correct, the intensity needs to be
            # corrected to raise the sum from 0.7 to 1.0 => Thus pixel value divided by 0.7.
            full_image = np.divide(
                full_image << 8, # (* 255 because the weights are represented as uint8 (1.0 <=> 255))
                weight_map.astype(np.uint64)
            ).astype(self.images_dtype)

            # Save the resulting image.
            path_result = os.path.join(self.img_path_base, 'result.png')

            # Remove result if exists
            try:
                os.remove(path_result)
            except:
                pass

            #np_background = cv2.equalizeHist(np_background)

            cv2.imwrite(path_result, full_image)

            # Clean individual images after result was saved if specified
            # Warning: Seems to be bugged on windows when filename is too long
            if clean_after:
                for filename in glob.glob(os.path.join(self.img_path_base, '*.png')):
                    basename = os.path.basename(filename)
                    if basename not in self.ignore_filenames:
                        os.remove(filename)

            print("...Done.")

    def coords_motor2cam(self, x, y):
        """
        Return tuple (x, y) of coordinates in cam referencial from motor referencial
        """
        theta = self.cam2motor_angle
        x2 = np.cos(theta) * x + np.sin(theta) * y
        y2 = - np.sin(theta) * x + np.cos(theta) * y
        return x2, y2

    def coords_cam2motor(self, x, y):
        """
        Return tuple (x, y) of coordinates in motor referencial from cam referencial
        """
        theta = self.cam2motor_angle
        x2 = np.cos(theta) * x - np.sin(theta) * y
        y2 = np.sin(theta) * x + np.cos(theta) * y
        return x2, y2
