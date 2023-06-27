import cv2
import numpy as np
from matplotlib import pyplot as plt

# ========================================= UNCOMMENT THE RIGHT LENS ====================================
#lens = "5"
lens = "20"
#lens = "50"
# =======================================================================================================

lut_file_name = f"x{lens}_vignette_lut.txt"

# Load reference blank image.
# If the saved pictures are grayscale that were converted to RGB, 1 channel
# can be used and the others discarded.
ref_img = cv2.imread(f"vignette_corr_refx{lens}.png", cv2.IMREAD_UNCHANGED)
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
lut = np.array(ref_img).astype(np.float64)

# In order to correct the uneven intensity distribution accross rows and columns,
# a ratio of correction is calculated for each pixel.
rows_mean = Y_rows.mean()
cols_mean = Y_cols.mean()
print(f"row_mean = {rows_mean} ; col_mean = {cols_mean}")
for col in range(W):
    for row in range(H):
        a = (float(rows_mean) / float(np.maximum(1, Y_rows[row])))
        b = (float(cols_mean) / float(np.maximum(1, Y_cols[col])))
        lut[row, col] = a * b

# Show a heatmap of the LUT.
plt.imshow(lut, cmap='hot', interpolation='nearest')
plt.show()

# Save generated lookup table in textfile as flattened array
np.savetxt(lut_file_name, lut.reshape((W * H)))

# ============================================ OPTIONAL =================================================

# Apply LUT on reference image as an example
image = cv2.imread(f"vignette_corr_refx{lens}.png", cv2.IMREAD_UNCHANGED)
image = image[1:1+H, 1:1+W]

image = image.astype(np.float64)
image = np.multiply(image, lut).astype(np.uint16)
#image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

# Save corrected reference image as an example
cv2.imwrite(f"vignette_corr_refx{lens}_corrected.png", image)

# =======================================================================================================
