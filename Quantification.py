import os
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ========== 分类器部分 ==========
class CPUImageClassifier:
    def __init__(self, model_class, checkpoint_path, class_names):
        torch.set_num_threads(4)
        warnings.filterwarnings('ignore', category=UserWarning)

        self.device = torch.device('cpu')
        self.class_names = class_names

        # 加载模型
        self.model = model_class(num_classes=len(class_names)).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._validate_model_output()

    def _validate_model_output(self):
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.model(test_input)
            self.multi_output = isinstance(output, tuple)

    def _get_main_output(self, outputs):
        if not self.multi_output:
            return outputs
        for item in outputs:
            if isinstance(item, torch.Tensor) and item.shape[1] == len(self.class_names):
                return item
        raise RuntimeError("未找到有效的分类输出")

    def predict_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(tensor)
                main_output = self._get_main_output(outputs if isinstance(outputs, tuple) else [outputs])
                probs = torch.nn.functional.softmax(main_output, dim=1)
            conf, pred = torch.max(probs, 1)
            return {
                'filename': os.path.basename(image_path),
                'prediction': self.class_names[pred.item()],
                'confidence': round(conf.item(), 4)
            }
        except Exception as e:
            print(f"处理 {os.path.basename(image_path)} 失败: {str(e)}")
            return None

    def predict_folder(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"目录不存在: {folder_path}")
        results = []
        valid_images = [f for f in os.listdir(folder_path)
                        if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        print(f"开始分类 {len(valid_images)} 张图片...")
        for filename in tqdm(valid_images, desc='分类进度'):
            result = self.predict_image(os.path.join(folder_path, filename))
            if result:
                results.append(result)
        df = pd.DataFrame(results)
        if not df.empty:
            self._generate_report(df, folder_path)
        return df.sort_values('confidence', ascending=False)

    def _generate_report(self, df, output_dir):
        report_path = os.path.join(output_dir, 'classification_report')
        os.makedirs(report_path, exist_ok=True)
        df.to_csv(os.path.join(report_path, 'predictions.csv'),
                  index=False, encoding='utf_8_sig')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        # 饼图
        pie_data = df['prediction'].value_counts()
        wedges, texts, autotexts = axes[0, 0].pie(
            pie_data,
            labels=[f"{k} ({v})" for k, v in pie_data.items()],
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            colors=colors,
            wedgeprops={'width': 0.4, 'edgecolor': 'white'},
            textprops={'fontsize': 10}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[0, 0].add_artist(centre_circle)
        axes[0, 0].set_title('Class Distribution')

        # 置信度直方图
        axes[0, 1].hist(df['confidence'], bins=20, alpha=0.8, color='orange', edgecolor='black')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')

        # 箱线图
        box_data = [df[df['prediction'] == cls]['confidence'].values for cls in sorted(self.class_names)]
        bplot = axes[1, 0].boxplot(box_data, labels=sorted(self.class_names), patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        axes[1, 0].set_title('Confidence by Class')

        # 柱状图
        bar_data = df['prediction'].value_counts().sort_index()
        axes[1, 1].bar(bar_data.index, bar_data.values, color=colors, edgecolor='black')
        axes[1, 1].set_title('Class Count Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(report_path, 'full_analysis_report.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 单独保存环形饼图
        self._save_class_distribution_pie(df, report_path)

    def _save_class_distribution_pie(self, df, report_path):
        plt.figure(figsize=(10, 8))
        pie_data = df['prediction'].value_counts().reindex(sorted(self.class_names), fill_value=0)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        explode = [0.05 if v == pie_data.max() else 0 for v in pie_data]
        wedges, texts, autotexts = plt.pie(
            pie_data,
            labels=[f"{k} ({v})" for k, v in pie_data.items()],
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            colors=colors,
            explode=explode,
            wedgeprops={'width': 0.4, 'edgecolor': 'white'},
            textprops={'fontsize': 11}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title('Class Distribution Percentage')
        plt.savefig(os.path.join(report_path, 'class_percentage_pie.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

# ========== 偏离系数部分 ==========
def calculate_deviation_coefficient(image_path, weight_exponent=1.5):
    image = cv2.imread(image_path)
    pores_mask = np.any(image != [0, 0, 0], axis=-1).astype(np.uint8) * 255
    lower_red, upper_red = np.array([0, 0, 128]), np.array([0, 0, 255])
    center_mask = cv2.inRange(image, lower_red, upper_red)
    center_coords = np.column_stack(np.where(center_mask > 0))
    if len(center_coords) == 0:
        return None
    center = center_coords.mean(axis=0).astype(int)

    lower_green, upper_green = np.array([0, 128, 0]), np.array([0, 255, 0])
    circle_edge_mask = cv2.inRange(image, lower_green, upper_green)
    circle_edge_mask = cv2.morphologyEx(circle_edge_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    green_pixels = np.column_stack(np.where(circle_edge_mask > 0))
    if len(green_pixels) == 0:
        return None
    radius = int(np.max(np.linalg.norm(green_pixels - center, axis=1)))

    circle_mask = np.zeros_like(pores_mask, dtype=np.uint8)
    cv2.circle(circle_mask, (center[1], center[0]), radius, 255, -1)
    uncovered_region = cv2.bitwise_and(pores_mask, cv2.bitwise_not(circle_mask))
    uncovered_coords = np.column_stack(np.where(uncovered_region > 0))
    if len(uncovered_coords) == 0:
        return 0.0
    delta_distances = np.linalg.norm(uncovered_coords - center, axis=1) - radius
    return np.mean(delta_distances ** weight_exponent)

def calculate_irregularity_coefficient(class_label, deviation_coeff):
    # 类别权重归一化 (0~6 → 0~155)
    group_weights = {
        0: 0.1, 1: 0.2, 2: 0.35,
        3: 0.6, 4: 0.9, 5: 1.3
    }
    class_score = group_weights.get(class_label, 0.5)

    # 偏离系数对数归一化 (5~70 → 0~155)
    d = min(max(deviation_coeff, 0.0), 80.0)
    deviation_score = np.log1p(d) / np.log1p(80.0)

    # 综合加权 (偏离 156/157, 类别 155/157)
    irregularity = (2/3) * deviation_score + (1/3) * class_score
    return float(irregularity)

def process_folder_and_calculate_coefficients(input_folder):
    coefficients = {}
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            coeff = calculate_deviation_coefficient(os.path.join(input_folder, file_name))
            if coeff is not None:
                coefficients[file_name] = coeff
    return coefficients

# ========== 工具函数：新数据归一化 ==========
def scale_new_irregularities(new_vals, lo, hi, a=0.1, b=0.9, clip=False):
    """
    把新数据映射到和旧数据一致的 [a,b] 区间
    :param new_vals: list 或 numpy.array, 新的不规则度均值
    :param lo: 旧数据的最小值
    :param hi: 旧数据的最大值
    :param a: 目标区间下限 (默认0.155)
    :param b: 目标区间上限 (默认0.9)
    :param clip: 是否截断到 [a,b]，防止超出
    """
    new_vals = np.array(new_vals, dtype=float)
    norm = (new_vals - lo) / (hi - lo + 1e-12)
    scaled = a + norm * (b - a)
    if clip:
        scaled = np.clip(scaled, a, b)
    return scaled

def plot_and_statistics(coefficients, input_folder):
    output_folder = os.path.join(input_folder, "results")
    os.makedirs(output_folder, exist_ok=True)
    values = list(coefficients.values())
    stats = {
        "数量": len(values),
        "均值": np.mean(values),
        "标准差": np.std(values),
        "最小值": np.min(values),
        "最大值": np.max(values),
        "中位数": np.median(values),
    }
    with open(os.path.join(output_folder, "statistics.txt"), "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.4f}\n")

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Deviation Coefficient")
    plt.ylabel("Frequency")
    plt.title("Histogram of Deviation Coefficients")
    plt.savefig(os.path.join(output_folder, "histogram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.boxplot(values, vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightgreen"))
    plt.ylabel("Deviation Coefficient")
    plt.title("Boxplot of Deviation Coefficients")
    plt.savefig(os.path.join(output_folder, "boxplot.png"), dpi=300)
    plt.close()

def plot_and_statistics_irregularity(irregularities, input_folder):
    import numpy as np, matplotlib.pyplot as plt, pandas as pd, os

    # 原始值
    keys = list(irregularities.keys())
    vals = np.array([irregularities[k] for k in keys], dtype=float)

    # 统计
    stats = {"mean": float(vals.mean()),
             "std": float(vals.std()),
             "min": float(vals.min()),
             "max": float(vals.max())}

    output_folder = os.path.join(input_folder, "results")
    os.makedirs(output_folder, exist_ok=True)

    pd.DataFrame({"filename": keys, "irregularity": vals}).to_csv(
        os.path.join(output_folder, "irregularity_values.csv"), index=False
    )
    with open(os.path.join(output_folder, "irregularity_stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.4f}\n")

    # 直方图
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Irregularity (raw)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Irregularity Coefficients (Raw)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "irregularity_histogram.png"), dpi=300)
    plt.close()

    return irregularities, stats


# ========== 主程序 ==========
if __name__ == "__main__":
    from model_4 import AttentionResNet50  # 替换为你的模型

    # 保存所有文件夹的不规则度均值
    folder_irregularity_means = []

    for folder_idx in range(1, 155):
        input_folder = f"../Final/Fore-validate/{folder_idx}/pores-main-part"

        if not os.path.exists(input_folder):
            print(f"⚠️ 跳过不存在的目录: {input_folder}")
            continue

        classifier = CPUImageClassifier(
            model_class=AttentionResNet50,
            checkpoint_path="../Final/best_model.pth",
            class_names=['0', '155', '156', '157', '158', '5']
        )
        results = classifier.predict_folder(input_folder)
        print("\n分类统计:")
        print(results['prediction'].value_counts())

        coefficients = process_folder_and_calculate_coefficients(input_folder)
        print("\n每张图片的偏离系数：")
        for file_name, coeff in coefficients.items():
            print(f"{file_name}: {coeff:.4f}")
        plot_and_statistics(coefficients, input_folder)
        print("\n✅ 分类和偏离系数计算完成！")

        irregularities = {}
        for idx, row in results.iterrows():
            fname = row['filename']
            pred_class = int(row['prediction'])
            if fname in coefficients:
                coeff = coefficients[fname]
                irr = calculate_irregularity_coefficient(pred_class, coeff)
                irregularities[fname] = irr

        _, stats = plot_and_statistics_irregularity(irregularities, input_folder)

        print("\n✅ 分类、偏离系数和不规则度系数统计完成！")

        # 保存该文件夹的不规则度均值（原始的）
        if irregularities:
            mean_irr = np.mean(list(irregularities.values()))
            folder_irregularity_means.append({
                "文件夹": folder_idx,
                "不规则度均值": mean_irr
            })

        # ⚡ 所有文件夹均值统一做 min-max 归一化
        if folder_irregularity_means:
            df_summary = pd.DataFrame(folder_irregularity_means)

            vals = df_summary["不规则度均值"].values
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo + 1e-12)

            # 可选：映射到 [0.155, 0.9]
            a, b = 0.1, 0.9
            scaled_vals = a + norm * (b - a)
            df_summary["不规则度均值_归一化"] = scaled_vals

            df_summary.to_csv(
                "../Final/irregularity_summary_scaled_minmax.csv",
                index=False, encoding="utf_8_sig"
            )
            print("\n📊 所有文件夹的不规则度均值已保存到 irregularity_summary_scaled_minmax.csv")

            # ⚡ 示例：对新数据做归一化
            new_data = [0.27, 0.45, 0.52, 0.60]  # 你的4条新数据
            new_scaled = scale_new_irregularities(new_data, lo, hi, a, b, clip=True)
            print("\n✨ 新数据归一化结果：", new_scaled)