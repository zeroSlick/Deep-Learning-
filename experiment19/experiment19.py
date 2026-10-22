import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms.functional import to_tensor

torch.manual_seed(0)

try:
    torch.hub.download_url_to_file("https://ultralytics.com/images/bus.jpg", "bus.jpg", progress=False)
    image = Image.open("bus.jpg").convert("RGB")
    truth = [22, 231, 805, 756]
    print("Image: sample bus.jpg (ground-truth box is a hand-drawn bus box for demonstration, not Pascal VOC)")
except Exception:
    from sklearn.datasets import load_sample_image
    print("FALLBACK: bus.jpg unavailable, using scikit-learn sample image; no ground-truth box available")
    image = Image.fromarray(load_sample_image("china.jpg"))
    truth = None

weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.5)
model.eval()
categories = weights.meta["categories"]

with torch.no_grad():
    result = model([to_tensor(image)])[0]

fig, ax = plt.subplots()
ax.imshow(image)
for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
    x1, y1, x2, y2 = box.tolist()
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2))
    ax.text(x1, y1, categories[label] + " " + str(round(float(score), 2)), color="white", backgroundcolor="red")
    print(categories[label], round(float(score), 2), [round(v) for v in box.tolist()])
ax.axis("off")
plt.show()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


bus_boxes = [b.tolist() for b, l in zip(result["boxes"], result["labels"]) if categories[l] == "bus"]
if truth is not None and bus_boxes:
    print("IoU of predicted bus box with ground-truth box:", round(iou(bus_boxes[0], truth), 4))
