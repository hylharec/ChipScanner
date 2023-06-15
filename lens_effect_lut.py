import cv2
import numpy as np
from matplotlib import pyplot as plt

lut_file_name = r"x50_vignette_lut.txt"

# Load reference blank image.
# If the saved pictures are grayscale that were converted to RGB, 1 channel
# can be used and the others discarded.
ref_img, _, _ = cv2.split(cv2.imread("vignetting_correction_ref/vignette_corr_refx50.png", cv2.IMREAD_UNCHANGED))
H, W = ref_img.shape

# Since the photocamera yields pictures with a 1 pixel wide black edge,
# it needs to be removed so as to not interfere with the computations.
ref_img = ref_img[1:H-1, 1:W-1]
H, W = ref_img.shape

# Creation of the row and column intensity sums.
# It shows how the grayscale intensity is distributed accross rows and columns.
Y_rows = np.sum(ref_img, axis=1)
Y_cols = np.sum(ref_img, axis=0)

# The vignette effect correction lookup table will contain percentage values,
# hence the use of float128 type.
lut = np.array(ref_img).astype(np.float128)

# In order to correct the uneven intensity distribution accross rows and columns,
# a ratio of correction is calculated for each pixel.
rows_mean = Y_rows.mean()
cols_mean = Y_cols.mean()
print(f"row_mean = {rows_mean} ; col_mean = {cols_mean}")
for col in range(W):
    for row in range(H):
        a = (float(rows_mean) / float(np.maximum(1, Y_rows[row])))
        b = (float(cols_mean) / float(np.maximum(1, Y_cols[col])))
        #print(a)
        lut[row, col] = a * b

# Uncomment the following lines to show a heatmap of the LUT.
plt.imshow(lut, cmap='hot', interpolation='nearest')
plt.show()

# Save generated lookup table in textfile as flattened array
np.savetxt(lut_file_name, lut.reshape((W * H)))

#lut = np.loadtxt(lut_file_name)
#lut = lut.reshape((H, W))

image,_,_ = cv2.split(cv2.imread("vignetting_correction_ref/vignette_corr_refx50.png", cv2.IMREAD_UNCHANGED))
image = image[1:1+H, 1:1+W]

image = image.astype(np.float128)
image = np.multiply(image, lut).astype(np.uint8)
#image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
cv2.imshow("corrected lens effect", image)
cv2.waitKey()
cv2.imwrite("vignetting_correction_ref/vignette_corr_refx50_corrected.png", image)
