import cv2
import numpy as np
import os
from skimage.measure import label


def remove_high_irregularity_areas(binary_img, contours, irregularities, threshold=10):
    # 遍历每个轮廓及其对应的不规则度
    for contour, irregularity in zip(contours, irregularities):
        if irregularity > threshold:
            # 使用黑色填充不规则度大于10的区域 (即孔隙)
            cv2.drawContours(binary_img, [contour], -1, (0), thickness=cv2.FILLED)
        # print(f"Irregularity: {irregularity}, Filling black: {irregularity > threshold}")
    return binary_img

def recalculate_irregularities(binary_img):
    # 重新查找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 重新计算不规则度
    return [calculate_irregularity(contour) for contour in contours]

def calculate_black_regions(binary_img):
    # 获取连通区域信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    black_regions = []
    # 从1开始忽略背景区域
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 40 or area > 2100:  # 如果区域面积小于80或者大于200，则过滤掉
            binary_img[labels == i] = 0
        else:
            black_regions.append(area)  # 保存符合条件的区域面积

    return black_regions

def calculate_irregularity(contour):  # 通过轮廓的周长和面积来衡量其与圆形的偏差。
    perimeter = cv2.arcLength(contour, True)  # 计算轮廓的周长。contour 是输入的轮廓，True 表示轮廓是闭合的
    area = cv2.contourArea(contour)  # 来计算轮廓的面积。
    if perimeter == 0:  # 为了避免除以零的错误。
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)  # 计算轮廓的圆形度（circularity）。圆形度是衡量一个形状与完美圆形的接近程度的指标。
    irregularity = (1 - circularity) * 20  # 将圆形度转换为不规则度。
    return round(max(0, min(irregularity, 100)), 2) # 确保不规则性的值被限制在 0 到 100 的范围内。


def resize_with_padding(img, target_size=(1024, 1024)):  # 将输入图像调整为指定的目标尺寸，同时通过添加黑色边框（填充）来保持原始图像的纵横比不变。

    h, w = img.shape[:2]  # 从输入的图像 img 中提取其高度 (h) 和宽度 (w)。
    scale = min(target_size[0] / h, target_size[1] / w)  # 根据目标高度和宽度选择缩放因子，确保图像保持纵横比不变。
    new_w, new_h = int(w * scale), int(h * scale)  # 根据缩放比例计算新的图像宽度 new_w 和高度 new_h。
    resized_img = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)  # 将图像调整为新的尺寸 (new_w, new_h)。使用 cv2.INTER_AREA 作为插值方法
    # 计算需要的填充大小
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left
    # 添加黑色边框填充
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def enhance_edges(image):
    # 使用拉普拉斯算子来增强图像边缘
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize= 1)
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换为8位图像
    # 将边缘增强图像与原始图像叠加，增强边缘
    enhanced_image = cv2.addWeighted(image, 3, laplacian, 0.6, 0)

    return enhanced_image


def has_excess_regions(image, max_regions=1):
    """
    检测图像中是否包含超过 max_regions 个连通区域。

    参数:
        image (numpy.ndarray): 二值化的输入图像，白色区域为目标（255），黑色为背景（0）。
        max_regions (int): 允许的最大连通区域数量。

    返回:
        bool: 如果连通区域数量超过 max_regions，返回 True；否则返回 False。
    """
    # 检测连通性


    white_labels, num_white = label(image == 255, connectivity=1, return_num=True, background=-1)
    black_labels, num_black = label(image == 0, connectivity=1, return_num=True, background=-1)
    num_regions = num_white + num_black  # 计算所有连通区域

    return num_regions > max_regions


def save_individual_pores(final_img, save_folder="pores", output_size=128, max_regions=1):
    global GLOBAL_PORE_COUNTER  # 声明使用全局计数器

    contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    pore_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped = final_img[y:y + h, x:x + w]

        # 添加边框和填充为正方形（代码不变）
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

        # 使用全局计数器生成唯一文件名
        GLOBAL_PORE_COUNTER += 1
        pore_file_path = os.path.join(save_folder, f"pore_{GLOBAL_PORE_COUNTER}.png")
        cv2.imwrite(pore_file_path, resized)
        pore_count += 1

    print(f"Saved {pore_count} pores (Total pores so far: {GLOBAL_PORE_COUNTER})")

def process_sem_image(image_path, threshold_method='manual', target_size=(1024, 1024)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式加载图像

    if img is None:
        print("Error: Image not found or could not be opened.")
        return

    # 创建保存图像的文件夹，文件夹名称为图像文件名（不带扩展名）
    folder_name = os.path.splitext(os.path.basename(image_path))[0]
    save_folder = save_folder = os.path.dirname(image_path)
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Folder created: {save_folder}")
    else:
        print(f"Folder already exists: {save_folder}")
    print(f"Original Image Shape: {img.shape}")
    resized_img = resize_with_padding(img, target_size)  # 保持比例调整图像大小并添加填充
    print(f"after_padding Image Shape: {resized_img.shape}")
    # cv2.imshow("1##.Grayscale image after filling", resized_img)
    # cv2.imwrite(os.path.join(save_folder, "1##.Grayscale_image_after_filling.png"), resized_img)
    saved_image_path = os.path.join(save_folder, "1##.Grayscale_image_after_filling.png")
    # cv2.imwrite(saved_image_path, resized_img)
    if os.path.exists(saved_image_path):
        print(f"Image saved successfully: {saved_image_path}")
    else:
        print(f"Failed to save image: {saved_image_path}")

    # equalized_img = cv2.equalizeHist(resized_img)  # 直方图均衡化，增强图像对比度
    enhanced_img = enhance_edges(resized_img)
    # cv2.imshow("3_NO.After enhancing the border", enhanced_img)
   # cv2.imwrite(os.path.join(save_folder, "3_NO.After_enhancing_the_border.png"), enhanced_img)
    # 根据选择的阈值方法进行二值化处理
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

    binary_img = cv2.bitwise_not(binary_img)  # 反转颜色

    # cv2.imshow("4#.After binarization", binary_img)
    #cv2.imwrite(os.path.join(save_folder, "4#.After_binarization.png"), binary_img)
    binary_img = cv2.medianBlur(binary_img, 5)  # 中值滤波器,5是滤波器的尺,
    # 作用是去除孔隙中间的失真值，并平滑受孔隙边缘失真造成的边缘锯齿形状。
    # cv2.imshow("5_有最大圆.After median filtering", binary_img)
    #cv2.imwrite(os.path.join(save_folder, "5_有最大圆.After_median_filtering.png"), binary_img)

    black_regions = calculate_black_regions(binary_img)
    print(f"孔的数量(去掉弱噪声后): {len(black_regions)}")
    print(f"孔的面积: {black_regions}")
    # cv2.imshow("6#.After removing the weak noise", binary_img)
   # cv2.imwrite(os.path.join(save_folder, "6#.After_removing_the_weak_noise.png"), binary_img)
    # 找到所有轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    irregularities = []
    for contour in contours:
        irregularity = calculate_irregularity(contour)
        irregularities.append(irregularity)

    print(f"irregularities", irregularities)
    final_img = remove_high_irregularity_areas(binary_img, contours, irregularities, threshold=13)

    # 重新计算移除后剩余区域的轮廓和不规则度
    new_irregularities = recalculate_irregularities(final_img)
    print(f"new_irregularities", new_irregularities)

    # 显示处理后的图像
    # cv2.imshow("7#.Final image after removing strong noise", final_img)
   # cv2.imwrite(os.path.join(save_folder, "7#.Final_image_after_removing_strong_noise.png"), final_img)
   #  cv2.waitKey(0)
   #  cv2.destroyAllWindows()

    return round(np.mean(new_irregularities)*2-9, 2), final_img

if __name__ == "__main__":
    # 1. 定义允许的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif')

    for folder_idx in range(1, 155):
        # 全局孔隙计数器
        GLOBAL_PORE_COUNTER = 0  # 初始化计数器

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

            # 检查文件扩展名
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue

            # 4. 处理图片
            print(f"正在处理图片: {filename}")
            try:
                threshold_method = 'manual'
                average_irregularity, final_img = process_sem_image(image_path, threshold_method=threshold_method)
                print(f"平均不规则度 ({filename}): {average_irregularity}")

                # 5. 保存结果到pores目录
                save_individual_pores(final_img, save_folder=save_folder)
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {str(e)}")

