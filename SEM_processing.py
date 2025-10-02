import cv2
import numpy as np
import os
from skimage.measure import label


def remove_high_irregularity_areas(binary_img, contours, irregularities, threshold=10):
    # Traverse each contour and its corresponding irregularity
    for contour, irregularity in zip(contours, irregularities):
        if irregularity > threshold:
            # Fill areas with an irregularity greater than 10 (i.e., pores) with black
            cv2.drawContours(binary_img, [contour], -1, (0), thickness=cv2.FILLED)
        # print(f"Irregularity: {irregularity}, Filling black: {irregularity > threshold}")
    return binary_img

def recalculate_irregularities(binary_img):
    # Re-find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Recalculate irregularity
    return [calculate_irregularity(contour) for contour in contours]

def calculate_black_regions(binary_img):
    # Get connected area information
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    black_regions = []
    # Ignore background area starting from 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 40 or area > 2100:  # If the area is less than 80 or greater than 200, filter it out
            binary_img[labels == i] = 0
        else:
            black_regions.append(area)  # Save the area of ​​the region that meets the conditions

    return black_regions

def calculate_irregularity(contour):  # The deviation from a circle is measured by the perimeter and area of ​​a contour
    perimeter = cv2.arcLength(contour, True)  # Calculates the perimeter of a contour. contour is the input contour, and True indicates that the contour is closed.
    area = cv2.contourArea(contour)  # Calculate the area of ​​a contour
    if perimeter == 0:  # To avoid division by zero errors
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)  # Calculate the circularity of the outline. Circularity is a measure of how close a shape is to being a perfect circle
    irregular = (1 - circularity) * 20  # Converts circularity to irregularity
    return round(max(0, min(irregular, 100)), 2) # Make sure the irregularity value is constrained to the range of 0 to 100


def resize_with_padding(img, target_size=(1024, 1024)):  # Resize the input image to the specified target size while maintaining the aspect ratio of the original image by adding a black border (padding).

    h, w = img.shape[:2]  # Extract the height (h) and width (w) of the input image img
    scale = min(target_size[0] / h, target_size[1] / w)  # Select a scaling factor based on the target height and width to ensure the image maintains its aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)  # Calculate the new image width new_w and height new_h based on the scaling factor
    resized_img = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)  # Resize the image to the new size (new_w, new_h). Use cv2.INTER_AREA as the interpolation method.
    # Calculate the required padding size
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left
    # Add black border fill
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def enhance_edges(image):
    #Use Laplacian operator to enhance image edges
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize= 1)
    laplacian = cv2.convertScaleAbs(laplacian)  
    enhanced_image = cv2.addWeighted(image, 3, laplacian, 0.6, 0)

    return enhanced_image


def has_excess_regions(image, max_regions=1):
    """
    Checks if an image contains more than max_regions connected regions.
    Parameters:
    image (numpy.ndarray): Binarized input image, with white regions representing the target (255) and black representing the background (0).
    max_regions (int): Maximum number of connected regions allowed.
    Returns:
    bool: Returns True if the number of connected regions exceeds max_regions; otherwise returns False.
    """

    white_labels, num_white = label(image == 255, connectivity=1, return_num=True, background=-1)
    black_labels, num_black = label(image == 0, connectivity=1, return_num=True, background=-1)
    num_regions = num_white + num_black  # Calculate all connected areas

    return num_regions > max_regions


def save_individual_pores(final_img, save_folder="pores", output_size=128, max_regions=1):
    global GLOBAL_PORE_COUNTER  

    contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    pore_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped = final_img[y:y + h, x:x + w]

        # Add a border
        border_size = 10
        padded = cv2.copyMakeBorder(cropped, border_size, border_size, border_size, border_size,
                                    cv2.BORDER_CONSTANT, value=0)
        max_side = max(padded.shape[:2])
        square_image = np.zeros((max_side, max_side), dtype=np.uint8)
        start_y = (max_side - padded.shape[0]) // 2
        start_x = (max_side - padded.shape[1]) // 2
        square_image[start_y:start_y + padded.shape[0], start_x:start_x + padded.shape[1]] = padded

        resized = cv2.resize(square_image, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

        if has_excess_regions(resized, max_regions=2):
            continue

        # Generate unique file names using a global counter
        GLOBAL_PORE_COUNTER += 1
        pore_file_path = os.path.join(save_folder, f"pore_{GLOBAL_PORE_COUNTER}.png")
        cv2.imwrite(pore_file_path, resized)
        pore_count += 1

    print(f"Saved {pore_count} pores (Total pores so far: {GLOBAL_PORE_COUNTER})")

def process_sem_image(image_path, threshold_method='manual', target_size=(1024, 1024)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale mode

    if img is None:
        print("Error: Image not found or could not be opened.")
        return

    # Create a folder to save images
    folder_name = os.path.splitext(os.path.basename(image_path))[0]
    save_folder = save_folder = os.path.dirname(image_path)
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Folder created: {save_folder}")
    else:
        print(f"Folder already exists: {save_folder}")
    print(f"Original Image Shape: {img.shape}")
    resized_img = resize_with_padding(img, target_size) 
    print(f"after_padding Image Shape: {resized_img.shape}")
    # cv2.imshow("1##.Grayscale image after filling", resized_img)
    # cv2.imwrite(os.path.join(save_folder, "1##.Grayscale_image_after_filling.png"), resized_img)
    saved_image_path = os.path.join(save_folder, "1##.Grayscale_image_after_filling.png")
    # cv2.imwrite(saved_image_path, resized_img)
    if os.path.exists(saved_image_path):
        print(f"Image saved successfully: {saved_image_path}")
    else:
        print(f"Failed to save image: {saved_image_path}")

    
    enhanced_img = enhance_edges(resized_img)
    # cv2.imshow("3_NO.After enhancing the border", enhanced_img)
    # cv2.imwrite(os.path.join(save_folder, "3_NO.After_enhancing_the_border.png"), enhanced_img)
    # Binarization is performed based on the selected threshold method
    if threshold_method == 'manual':
        manual_thresh_value = 42
        _, binary_img = cv2.threshold(enhanced_img, manual_thresh_value, 255, cv2.THRESH_BINARY)
        # "Manual Threshold Applied"
    elif threshold_method == 'adaptive':
        binary_img = cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # "Step 5_有最大圆: Adaptive Threshold Applied"
    else:
        _, binary_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # "Step 5_有最大圆: Otsu Threshold Applied"

    binary_img = cv2.bitwise_not(binary_img)  

    # cv2.imshow("4#.After binarization", binary_img)
    #cv2.imwrite(os.path.join(save_folder, "4#.After_binarization.png"), binary_img)
    binary_img = cv2.medianBlur(binary_img, 5)  
    # cv2.imshow("5_有最大圆.After median filtering", binary_img)
    #cv2.imwrite(os.path.join(save_folder, "5_有最大圆.After_median_filtering.png"), binary_img)

    black_regions = calculate_black_regions(binary_img)
    print(f"孔的数量(去掉弱噪声后): {len(black_regions)}")
    print(f"孔的面积: {black_regions}")
    # cv2.imshow("6#.After removing the weak noise", binary_img)
    # cv2.imwrite(os.path.join(save_folder, "6#.After_removing_the_weak_noise.png"), binary_img)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    irregularities = []
    for contour in contours:
        irregularity = calculate_irregularity(contour)
        irregularities.append(irregularity)

    print(f"irregularities", irregularities)
    final_img = remove_high_irregularity_areas(binary_img, contours, irregularities, threshold=13)

    new_irregular = recalculate_irregularities(final_img)
    print(f"new_irregular", new_irregular)

    # 显示处理后的图像
    # cv2.imshow("7#.Final image after removing strong noise", final_img)
    # cv2.imwrite(os.path.join(save_folder, "7#.Final_image_after_removing_strong_noise.png"), final_img)
    #  cv2.waitKey(0)
    #  cv2.destroyAllWindows()

    return round(np.mean(new_irregularities)*2-9, 2), final_img

if __name__ == "__main__":
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif')

    for folder_idx in range(1, 155):
        GLOBAL_PORE_COUNTER = 0  

        image_dir = f"../Final/FinalData-DealImage/{folder_idx}"

        if not os.path.exists(image_dir):
            print(f"⚠️ 跳过不存在的目录: {image_dir}")
            continue

        save_folder = os.path.join(image_dir, "pores")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"已创建输出目录: {save_folder}")

        pore_counter = 0
        for filename in os.listdir(image_dir):

            image_path= os.path.join(image_dir, filename)
            if not os.path.isfile(image_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue

            print(f"正在处理图片: {filename}")
            try:
                threshold_method = 'manual'
                average_irregularity, final_img = process_sem_image(image_path, threshold_method=threshold_method)
                print(f"平均不规则度 ({filename}): {average_irregularity}")

                save_individual_pores(final_img, save_folder=save_folder)
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {str(e)}")


