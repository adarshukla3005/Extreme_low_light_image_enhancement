import numpy as np
import cv2

def calculate_psnr(ref_image_path, dist_image_path):
    # Load the reference and distorted images
    ref_image = cv2.imread(ref_image_path)
    dist_image = cv2.imread(dist_image_path)

    # Convert images to grayscale if needed
    ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    dist_image_gray = cv2.cvtColor(dist_image, cv2.COLOR_BGR2GRAY)

    # Compute MSE
    mse = np.mean((ref_image_gray - dist_image_gray) ** 2)

    if mse == 0:
        # PSNR is infinity if mse is zero
        psnr = 100
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

# Example usage:
ref_image_path = 'test/low/12.jpg'
dist_image_path = 'test/predicted/processed_12.jpg'

psnr_value = calculate_psnr(ref_image_path, dist_image_path)
print(f'The PSNR value is: {psnr_value:.2f} dB')