import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from syn_dataset import DuckDataset, collate_fn

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

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(img)
                ax2.imshow(img)

                # Plot predictions on the left subplot
                for box, label in zip(output['boxes'], output['labels']):
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    color = 'b' if label == 1 else 'm'
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
                    ax1.add_patch(rect)

                # Plot ground truth on the right subplot
                for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    color = 'b' if label == 1 else 'm'
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
                    ax2.add_patch(rect)

                handles = [plt.Line2D([0], [0], color='b', lw=2, label='Male Duck'),
                           plt.Line2D([0], [0], color='m', lw=2, label='Female Duck')]
                ax1.legend(handles=handles)
                ax2.legend(handles=handles)

                ax1.set_title('Predictions')
                ax2.set_title('Ground Truth')

                plt.show()

                # for pred_box in output['boxes']:
                #     for gt_box in targets[i]['boxes']:
                #         iou = calculate_iou(pred_box.cpu().numpy(), gt_box.cpu().numpy())
                #         print(f"IoU: {iou}")