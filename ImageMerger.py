import os
import glob
import cv2
import numpy as np

class ImageMerger:
    def __init__(
        self,
        img_path_base = "img_to_merge",
        pxl_per_um: float = 640,
        lens: int = 50,
        merge_img_size = [1500, 1500], # [W, H] in pixels, should take margins into account
        margins_um = [200, 200], # [margin_x, margin_y]
        ignored_filenames = ["result.png", "result_np.png", "result1.png", "result2.png", "result3.png", "result4.png"],
        x_end_um = 100,
        y_end_um = 100,
        cam2motor_angle = 0.020 # in rad
    ):
        self.img_path_base = img_path_base
        self.pxl_per_um = pxl_per_um
        self.lens = lens
        self.merge_img_size = merge_img_size
        self.margins_um = margins_um
        self.ignored_filenames = ignored_filenames
        self.x_end_um = x_end_um
        self.y_end_um = y_end_um
        self.cam2motor_angle = cam2motor_angle # in rad

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

                raw_x = float(coords.split("_")[2]) + float(self.margins_um[0])
                raw_y = float(coords.split("_")[3]) + float(self.margins_um[1])

                # Convert coordinate system to try to compensate
                # rotation difference between camera and motor axis.
                x, y = self.coords_motor2cam(raw_x, raw_y)

                #x = self.x_end_um - x # x axis needs to be inverted

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
            print("Merge loaded images...")
            full_img_W, full_img_H = self.merge_img_size[0], self.merge_img_size[1]
            img_W, img_H = self.images_res[0], self.images_res[1]

            # Blank image used as background.
            np_background = np.zeros((full_img_H + 100, full_img_W + 100), dtype=self.images_dtype)
            bg_weights = np.zeros((full_img_H + 100, full_img_W + 100), dtype=self.images_dtype)
            weight = np.zeros((full_img_H, full_img_W), dtype=self.images_dtype)

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
            vignette_corr_lut = vignette_corr_lut.reshape((img_H, img_W))

            # Merge overlaping images.
            index = 0
            for x, y, image in self.images:
                print(f"Merging img ({x}, {y})...")
                # coords are micrometers
                x_px = int(np.round(x * self.pxl_per_um))
                y_px = int(np.round(y * self.pxl_per_um))

                # Apply vignette effect correction
                image = image.astype(np.float128)
                image[1:img_H-1, 1:img_W-1] = np.multiply(image[1:img_H-1, 1:img_W-1], vignette_corr_lut)
                image = image.astype(self.images_dtype)

                #gray_image,_,_,_ = cv2.split(np.array(image))

                # To apply the elliptic mask on the current image, a higher dtype is casted to prevent
                # overflow during computation. The the masked image is recasted into the correct dtype.
                masked_image = np.multiply(
                    #gray_image[1:img_H-1, 1:img_W-1],
                    image[1:img_H-1, 1:img_W-1].astype(np.uint64),
                    weight[1:img_H-1, 1:img_W-1].astype(np.uint64)
                ) >> 8 # (// 255 because the weights are represented as uint8 (1.0 <=> 255))
                masked_image = masked_image.astype(self.images_dtype)

                np_background[y_px+1:y_px+1+img_H-2, x_px+1:x_px+1+img_W-2] += masked_image
                bg_weights[   y_px+1:y_px+1+img_H-2, x_px+1:x_px+1+img_W-2] += weight[1:img_H-1, 1:img_W-1]

                #index += 1
                if index > 100:
                    break

            # Cells == 0 => no pixel was added, weight can be set to 1.0 without consequence (to prevent division by zero on next step)
            bg_weights[bg_weights == 0] = 255

            # Once all images were merged, apply a per pixel intensity correction depending on where the sum of weights
            # of overlaping pixels does not add up to 1.
            # For exemple, if one pixel is on the overlap of 3 images with weights of respectively 0.1, 0.4 and 0.2,
            # the sum only adds up to 0.7. Thus for the averaging process to be correct, the intensity needs to be
            # corrected to raise the sum from 0.7 to 1.0 => Thus pixel value divided by 0.7.
            np_background = np.divide(
                np_background.astype(np.uint64) << 8, # (* 255 because the weights are represented as uint8 (1.0 <=> 255))
                bg_weights.astype(np.uint64)
            ).astype(self.images_dtype)

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
