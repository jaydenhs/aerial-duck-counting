import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from syn_dataset import DuckDataset, collate_fn
from PIL import Image

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

if __name__ == '__main__':
    # Dataset and DataLoader
    dataset = DuckDataset('synthesized/', 'synthesized/annotations.json')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    model = fasterrcnn_resnet50_fpn()
    num_classes = 3  # 2 classes (male, female) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the trained model
    checkpoint = "20241107_155644_epoch200_loss0.0605"
    model.load_state_dict(torch.load(f'syn_checkpoints/{checkpoint}.pth', weights_only=True))
    model.eval()

    # Evaluation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Load the specific image
    image_path = r'C:\Users\Jayden Hsiao\Documents\Grad School\Projects\aerial-duck-counting\processed-data\images\d8d096d5-overhead_2.png_2_2.png'
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        output = outputs[0]

        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        ax1.imshow(img)

        # Plot predictions
        for box, label in zip(output['boxes'], output['labels']):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            color = 'b' if label == 1 else 'm'
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)

        handles = [plt.Line2D([0], [0], color='b', lw=2, label='Male Duck'),
                   plt.Line2D([0], [0], color='m', lw=2, label='Female Duck')]
        ax1.legend(handles=handles)

        ax1.set_title('Predictions')

        plt.show()
