import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib
from skimage import measure
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../Final/dataset_4"
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 10
LEARNING_RATES = {'base': 3e-4, 'head': 1e-3}
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
SAVE_DIR = "../Final/model_4-results"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "attention_maps"), exist_ok=True)

class RobustDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train'):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.phase = phase
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self._validate_samples()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        samples.append(item)
        return samples

    def _validate_samples(self):
        valid_samples = []
        for path, target in self.samples:
            if os.path.exists(path):
                valid_samples.append((path, target))
            else:
                print(f"警告: 文件 {path} 不存在，已跳过")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)

                att_target, inner_radius = self._generate_pore_attention_target(img)
                return img, target, att_target, inner_radius
        except Exception as e:
            print(f"加载图像 {path} 失败: {str(e)}")
            dummy_img = torch.zeros(3, 224, 224)
            dummy_att = torch.zeros(1, 224, 224)
            return dummy_img, target, dummy_att

    def _generate_pore_attention_target(self, img_tensor):
        """Generates precise circular target areas"""
        # Convert to PIL Image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img_pil = Image.fromarray(img_np.astype('uint8'))

        hsv = img_pil.convert('HSV')
        hsv_np = np.array(hsv)

        H, W = hsv_np.shape[0], hsv_np.shape[1]  # Get image height and width
        center = np.array([W // 2, H // 2])  # Coordinates of the geometric center of the image [x, y]

        green_mask = (hsv_np[..., 0] > 70) & (hsv_np[..., 0] < 160) & (hsv_np[..., 1] > 80)

        Y, X = np.ogrid[:H, :W]
        dist_map = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        max_radius = np.max(dist_map[green_mask]) if np.any(green_mask) else min(H, W) // 2 #The distance to the farthest point within the green area (if there is no green area, the default value is 1/156 of the short side of the image)
        inner_radius = max_radius * 0.75 # 80% of the maximum radius, forming a ring with a width of 10%

        inner_circle = dist_map <= inner_radius

        white_mask = (img_np.mean(axis=2) > 200 / 255)

        target_mask = white_mask & (~inner_circle)

        target_mask = gaussian_filter(target_mask.astype(float), sigma=5)
        target_mask = (target_mask - target_mask.min()) / (target_mask.max() - target_mask.min() + 1e-6)

        return torch.from_numpy(target_mask).float(), inner_radius

def prepare_data_loaders():

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RobustDataset(
        root_dir=os.path.join(DATA_DIR, 'train'),
        transform=train_transform,
        phase='train'
    )

    val_dataset = RobustDataset(
        root_dir=os.path.join(DATA_DIR, 'val'),
        transform=val_transform,
        phase='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, train_dataset.classes, val_transform


class AttentionResNet50(nn.Module):
    def __init__(self, num_classes, temperature=0.3):  # Temperature parameters
        super().__init__()
        self.temperature = temperature  # Temperature coefficient initialization

        # Backbone Network
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-2])

        # Attention Network
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, groups=512),  # Depthwise Separable Convolution
            nn.Conv2d(512, 256, 1),
            self._TemperatureModule(self.temperature),  # Insert the temperature module
            nn.Sigmoid(),
            nn.Conv2d(256, 1, 1),
            nn.Hardtanh(min_val=0.1, max_val=0.9)
        )

        # Inhibitory Network
        self.suppress_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(2048, num_classes)

    class _TemperatureModule(nn.Module):
        """Temperature scaling module (internal class)"""

        def __init__(self, temperature):
            super().__init__()
            self.temperature = temperature

        def forward(self, x):
            return x / self.temperature  

    def _create_dynamic_mask(self, inner_radius, device):
        """Generates a ring mask based on the given inner_radius"""
        B = len(inner_radius)
        H, W = 224, 224

        # Generate grid coordinates
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        dist = X ** 2 + Y ** 2  # [H, W]

        # Convert inner_radius to normalized radius
        normalized_radius = (inner_radius / 112.0).to(device)  
        normalized_radius = normalized_radius.view(B, 1, 1)  # [B, 155##, 155##]

        # Generate annular mask
        inner_mask = dist < normalized_radius ** 2
        outer_mask = dist < (normalized_radius * 1.1) ** 2  
        ring_mask = outer_mask & (~inner_mask)

        return ring_mask.float().unsqueeze(1)  # [B, 155##, H, W]

    def forward(self, x, inner_radius=None):
        features = self.backbone(x)

        # Generate dynamic masks
        if inner_radius is None:
            inner_radius = torch.full((x.size(0),), 20.0, device=x.device)
        dynamic_mask = self._create_dynamic_mask(inner_radius, x.device)

        # Generating Attention with Temperature Scaling
        attn = self.attention(features) 
        up_attn = F.interpolate(attn, size=224, mode='bilinear')

        # Apply dynamic mask (keep original)
        suppressed_attn = up_attn * (0.3 + 0.7 * dynamic_mask)

        # Classification output
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        out = self.fc(pooled.view(pooled.size(0), -1))

        return out, suppressed_attn.squeeze(1)

class Trainer:
    def __init__(self, model, train_loader, val_loader, val_dataset,
                 class_names, device, save_dir='saved_models', boundary_weight=3.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.device = device
        self.save_dir = save_dir
        self.boundary_weight = boundary_weight

        self.criterion = {
            'cls': nn.CrossEntropyLoss(),
            'att': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
        }

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'att_loss': [],
            'att_iou': [],
            'cls_loss': [],
            'edge_loss': []
        }

        self.attention_weight = 0.7

        self.optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': model.attention.parameters(), 'lr': 1e-3},
            {'params': model.suppress_net.parameters(), 'lr': 1e-3},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[1e-4, 1e-3, 1e-3, 1e-3],  
            steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS,
            pct_start=0.3
        )

        self.best_acc = 0.0
        self.best_model_wts = None
        self.boundary_weight = 3.0

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs('results/attention_maps', exist_ok=True)

    def _generate_weight_map(self, target):

        kernel = torch.ones(3, 3, device=target.device)
        dilated = F.conv2d(target.unsqueeze(1), kernel, padding=1) > 0
        eroded = F.conv2d(target.unsqueeze(1), kernel, padding=1) == 9
        boundary = (dilated.float() - eroded.float()).squeeze(1)

        weights = torch.ones_like(target)
        weights[boundary > 0] = self.boundary_weight
        return weights

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        cls_loss_sum = 0.0
        att_loss_sum = 0.0
        edge_loss_sum = 0.0

        current_suppress_strength = min(0.4 + epoch / num_epochs * 0.4, 0.8)

        for batch_idx, (inputs, targets, att_target, inner_radius) in enumerate(self.train_loader):

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            att_target = att_target.to(self.device, non_blocking=True)
            inner_radius = inner_radius.to(device) 
            with torch.no_grad():
                B, H, W = att_target.shape

                Y, X = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=self.device),
                    torch.linspace(-1, 1, W, device=self.device),
                    indexing='ij'
                )
                dist = X ** 2 + Y ** 2  # [H,W]

                base_weights = torch.where(
                    dist < current_suppress_strength,
                    0.2,
                    1.0  )

                kernel = torch.ones(3, 3, device=self.device)
                dilated = F.conv2d(
                    att_target.unsqueeze(1),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                ) > 0
                eroded = F.conv2d(
                    att_target.unsqueeze(1),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                ) == kernel.numel()
                boundary = (dilated.float() - eroded.float()).squeeze(1)

                weights = 0.7*base_weights + 0.3*(1 + self.boundary_weight*boundary)
                weights = weights.unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            outputs, pred_att = self.model(inputs, inner_radius)  

            pred_att = pred_att.unsqueeze(1)
            att_target = att_target.unsqueeze(1)

            att_loss = F.binary_cross_entropy(
                pred_att,
                att_target,
                weight=weights,
                reduction='mean'
            ) * self.attention_weight

            edge_kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9
            pred_edges = F.conv2d(pred_att, edge_kernel, padding=1)
            target_edges = F.conv2d(att_target, edge_kernel, padding=1)
            edge_loss = F.mse_loss(pred_edges, target_edges) * 0.3

            cls_loss = self.criterion['cls'](outputs, targets)

            total_loss = cls_loss + att_loss + edge_loss
            total_loss += 0.001 * torch.norm(pred_att, p=2)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=2.0,
                norm_type=2
            )

            self.optimizer.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            train_loss += total_loss.item()
            cls_loss_sum += cls_loss.item()
            att_loss_sum += att_loss.item()
            edge_loss_sum += edge_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {total_loss.item():.4f} (Cls: {cls_loss.item():.4f} "
                    f"Att: {att_loss.item():.4f} Edge: {edge_loss.item():.4f}) | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = train_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        self.history['cls_loss'].append(cls_loss_sum / len(self.train_loader))
        self.history['att_loss'].append(att_loss_sum / len(self.train_loader))
        self.history['edge_loss'].append(edge_loss_sum / len(self.train_loader))

        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        iou_scores = []

        with torch.no_grad():
            for images, labels, att_target, inner_radius in self.val_loader:
                # Data is moved to the device (including inner_radius)
                images = images.to(self.device)
                labels = labels.to(self.device)
                att_target = att_target.to(self.device)
                inner_radius = inner_radius.to(self.device)  

                # Forward propagation (passing in inner_radius)
                outputs, attentions = self.model(images, inner_radius)  

                # Calculating classification loss
                loss = self.criterion['cls'](outputs, labels)
                val_loss += loss.item()

                # Calculating IoU
                pred_mask = (attentions.squeeze(1) > 0.5).float()
                intersection = (pred_mask * att_target).sum(dim=[1, 2])
                union = (pred_mask + att_target).clamp(0, 1).sum(dim=[1, 2])
                iou_scores.extend((intersection / (union + 1e-6)).cpu().tolist())

                # Calculate classification accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if epoch % 3 == 0:
                self._visualize_attention(epoch, images, attentions, labels, att_target)

        # Statistical indicators
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        avg_iou = np.mean(iou_scores)

        # Learning rate scheduling and recording history
        self.scheduler.step(accuracy)
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        self.history['att_iou'].append(avg_iou)

        return avg_loss, accuracy

    def _visualize_attention(self, epoch, images, attentions, labels, att_target):  
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        selected = np.random.choice(len(images), 4, replace=False)

        for i, idx in enumerate(selected):
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

            pred_attn = attentions[idx].squeeze().cpu().numpy()
            vmin = np.percentile(pred_attn, 5)
            vmax = np.percentile(pred_attn, 95)
            pred_attn = np.clip((pred_attn - vmin) / (vmax - vmin), 0, 1)

            true_target = att_target[idx].squeeze().cpu().numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Original: {self.class_names[labels[idx]]}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_target, cmap='viridis')
            axes[i, 1].set_title("Target Region")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_attn, cmap='jet')
            axes[i, 2].set_title("Predicted Attention")
            axes[i, 2].axis('off')

            axes[i, 3].imshow(img)
            contours = measure.find_contours(pred_attn > 0.7)
            for contour in contours:
                axes[i, 3].plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            target_contours = measure.find_contours(true_target > 0.5)
            for contour in target_contours:
                axes[i, 3].plot(contour[:, 1], contour[:, 0], linewidth=1, color='lime', linestyle='--')
            axes[i, 3].set_title("Overlay (Red=Pred, Green=Target)")
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/attention_maps/epoch_{epoch}_enhanced.png", dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, num_epochs=50, patience=5):  
        # Initializing variables
        patience_counter = 0
        best_val_loss = float('inf')  

        # Make sure the history record exists with the required key
        self.history.setdefault('train_loss', [])
        self.history.setdefault('val_loss', [])
        self.history.setdefault('val_acc', [])
        self.history.setdefault('att_iou', [])

        # Learning rate scheduler (assuming optimizer is defined)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=2, verbose=True
        )

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()  # Make sure the model is in training mode
            train_loss = self.train_epoch(epoch, num_epochs)
            self.history['train_loss'].append(train_loss)  # Record training loss

            # Verification phase
            self.model.eval()  # Switch to evaluation mode
            val_loss, val_acc = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Update learning rate
            scheduler.step(val_acc)  # Adjust the learning rate based on the validation accuracy

            # Print log (make sure att_iou is recorded)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Att IoU: {self.history['att_iou'][-1]:.4f} | "  # Assume that the record is in validate
                f"Best Acc: {self.best_acc:.2f}%"
            )

            # Save the best model (taking both accuracy and loss into account)
            if val_acc > self.best_acc or (val_acc == self.best_acc and val_loss < best_val_loss):
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                best_val_loss = min(val_loss, best_val_loss)

                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_acc': self.best_acc,
                        'val_loss': val_loss,
                    },
                    os.path.join(self.save_dir, '../best_model.pth')
                )
                patience_counter = 0  
            else:
                patience_counter += 1
                if patience_counter >= patience:  
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Training end processing
        print(f"Training complete. Best Accuracy: {self.best_acc:.2f}%")
        self.model.load_state_dict(self.best_model_wts)  Load the best weights
        return self.history

if __name__ == "__main__":
    train_loader, val_loader, class_names, val_transform = prepare_data_loaders()
    print(f"Dataset loaded with {len(class_names)} classes: {class_names}")

    model = AttentionResNet50(len(class_names)).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    val_dataset = RobustDataset(
        root_dir=os.path.join(DATA_DIR, 'val'),
        transform=val_transform,
        phase='val'
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        class_names=class_names,
        device=device,
        save_dir=SAVE_DIR,
        boundary_weight=3.0
    )

    trainer.model.suppress_net[0].stride = (4, 4)
    trainer.attention_weight = 0.7

    test_input = torch.randn(2, 3, 224, 224).to(device)
    test_out, test_att = model(test_input)
    print(f"Test attention shape: {test_att.shape}")

    sample_batch = next(iter(trainer.train_loader))
    inputs, targets, att_target, inner_radius = sample_batch
    inputs, targets, att_target = inputs.to(device), targets.to(device), att_target.to(device)

    print("\n=== 维度检查 ===")
    print(f"输入图像形状: {inputs.shape} (应为 [B,3,224,224])")
    print(f"注意力目标形状: {att_target.shape} (应为 [B,224,224])")

    with torch.no_grad():
        test_out, test_att = trainer.model(inputs)
        print(f"\n模型分类输出形状: {test_out.shape} (应为 [B,num_classes])")
        print(f"模型注意力输出形状: {test_att.shape} (应为 [B,224,224])")

    assert test_att.shape == att_target.shape, f"维度不匹配! 注意力输出: {test_att.shape}, 目标: {att_target.shape}"
    print("=== 维度检查通过 ===\n")

    history = trainer.train(NUM_EPOCHS)

    best_model_path = os.path.join(SAVE_DIR, '../best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    with torch.no_grad():

        sample = next(iter(val_loader))
        print(len(sample))
        images, labels, att_target, *_ = sample

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        att_target = att_target.to(device, non_blocking=True)

        _, attentions = model(images)

        images = images.cpu()
        labels = labels.cpu()
        att_target = att_target.cpu()
        attentions = attentions.cpu()

    print("\n=== 可视化前维度检查 ===")
    print(f"图像维度: {images.shape} (应含 [B,3,224,224])")
    print(f"注意力图维度: {attentions.shape} (应含 [B,224,224])")
    print(f"目标掩码维度: {att_target.shape} (应含 [B,224,224])")

    trainer._visualize_attention(
        'final',
        images,
        attentions.unsqueeze(1) if attentions.dim() == 3 else attentions,  
        labels,
        att_target

    )
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), bbox_inches='tight')

    plt.close()
