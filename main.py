import torch
import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from models import CustomFasterRCNN
from utils import visualize_prediction

plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
# plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
# plt.rcParams['font.family'] = 'NanumGothic' # Linux의 경우

plt.rcParams['axes.unicode_minus'] = False


class Args:
    def __init__(self):
        # Data paths
        self.image_dir = "./data/ai04-level1-project/test_images"
        self.model_path = "./checkpoints/final_model.pth"
        self.label2name = './label2name.json'
        self.label2id = './label2id.json'

        # Inference parameters
        self.batch_size = 4
        self.num_workers = 4
        self.confidence_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # Model parameters
        self.num_classes = 74  # 반드시 모델 학습 시 사용한 클래스 수로 맞춰야 함

        # Output settings
        self.save_predictions = True
        self.prediction_output_path = "./predictions"
        self.save_visualizations = True
        self.visualization_output_path = "./visualizations"
        self.vis_num_samples = 20


def get_inference_transforms():
    """Inference transform"""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


class ImageOnlyDataset(Dataset):
    """이미지만 로드하는 Dataset"""
    def __init__(self, image_dir, transforms=None):
        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            self.image_paths.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(ext.split("*")[-1])])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, img_path


def load_model(model_path, num_classes, device):
    """모델 로드"""
    model = CustomFasterRCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def run_inference(model, data_loader, device, confidence_threshold=0.5):
    """데이터셋 전체 추론"""
    model.eval()
    all_predictions = []
    all_img_paths = []

    with torch.no_grad():
        for batch_idx, (images, img_paths) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, path in zip(outputs, img_paths):
                keep = pred["scores"] > confidence_threshold
                filtered = {
                    "boxes": pred["boxes"][keep].cpu(),
                    "labels": pred["labels"][keep].cpu(),
                    "scores": pred["scores"][keep].cpu()
                }
                all_predictions.append(filtered)
                all_img_paths.append(path)

    return all_predictions, all_img_paths


def save_predictions_to_json(predictions, image_paths, class_names, output_path, label2id_dict):
    """예측 결과 JSON 저장"""
    results = []
    for pred, img_path in zip(predictions, image_paths):
        image_id = os.path.basename(img_path).split('.')[0]
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min
            label = label2id_dict[str(label.item())]
            result = {
                "image_id": image_id,
                "category_id": int(label),
                "category_name": class_names[int(label)] if int(label) < len(class_names) else "unknown",
                "bbox": [x_min, y_min, width, height],
                "score": float(score)
            }
            results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to {output_path}")


def visualize_and_save_results(model, dataset, predictions, image_paths, class_names, args):
    """결과 시각화 및 저장"""
    os.makedirs(args.visualization_output_path, exist_ok=True)

    # 랜덤 샘플 선택
    indices = random.sample(range(len(dataset)), min(args.vis_num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        image_tensor, img_path = dataset[idx]
        image = image_tensor.permute(1, 2, 0).numpy()

        pred = predictions[idx]
        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        # 예측 결과 그리기
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min
            rect = plt.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor="red", facecolor="none")
            plt.gca().add_patch(rect)
            plt.text(
                x_min, y_min - 10,
                f"{class_names[label.item()]}: {score:.2f}",
                color="red", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7)
            )

        plt.title(f"Sample {i+1} - Predictions")
        plt.axis("off")
        plt.tight_layout()

        save_path = os.path.join(args.visualization_output_path, f"sample_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Visualization saved: {save_path}")


def main():
    args = Args()
    print(f"Using device: {args.device}")

    os.makedirs(args.prediction_output_path, exist_ok=True)

    # 모델 로드
    model = load_model(args.model_path, args.num_classes, args.device)

    # Dataset & DataLoader
    dataset = ImageOnlyDataset(args.image_dir, transforms=get_inference_transforms())
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # class_names = [f"class_{i}" for i in range(args.num_classes)]
    # class_names = args.label2name
    with open(args.label2name, 'r', encoding='utf-8') as f:
        label2name_dict = json.load(f)
    class_names = ['background'] + [name for name in label2name_dict.values()]
    print(class_names)
    # 추론 실행
    predictions, image_paths = run_inference(model, data_loader, args.device, args.confidence_threshold)
    print(predictions[0])

    with open(args.label2id, 'r', encoding='utf-8') as f:
        label2id_dict = json.load(f)
    # JSON 저장
    if args.save_predictions:
        output_json = os.path.join(args.prediction_output_path, "test_predictions.json")
        save_predictions_to_json(predictions, image_paths, class_names, output_json, label2id_dict=label2id_dict)

    # 시각화
    if args.save_visualizations:
        visualize_and_save_results(model, dataset, predictions, image_paths, class_names, args)

    print("Inference completed!")


if __name__ == "__main__":
    main()
