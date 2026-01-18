import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

import torchvision.models as models
import torchvision.transforms as transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ensemble models for explanation
model_1 = models.resnet50(weights=None)
model_1.fc = torch.nn.Linear(model_1.fc.in_features, 5)
model_1.load_state_dict(torch.load("model/best_model_1.pth", map_location=device, weights_only=True))
model_1 = model_1.to(device)
model_1.eval()

model_2 = models.densenet121(weights=None)
model_2.classifier = torch.nn.Linear(model_2.classifier.in_features, 5)
model_2.load_state_dict(torch.load("model/best_model_2.pth", map_location=device, weights_only=True))
model_2 = model_2.to(device)
model_2.eval()

model_3 = models.efficientnet_b0(weights=None)
model_3.classifier[1] = torch.nn.Linear(model_3.classifier[1].in_features, 5)
model_3.load_state_dict(torch.load("model/best_model_3.pth", map_location=device, weights_only=True))
model_3 = model_3.to(device)
model_3.eval()

ensemble_models = [model_1, model_2, model_3]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load image
# image_path = "C:\\Users\\amirt\\OneDrive\\Desktop\\finalproject\\Lung Disease Dataset\\test\\Corona Virus Disease\\_21_3861426.jpeg"
# image_pil = Image.open(image_path).convert("RGB")
# input_tensor = transform(image_pil).unsqueeze(0).to(device)

# # Get predicted class from ensemble
# with torch.no_grad():
#     outputs = [model(input_tensor) for model in ensemble_models]
#     avg_logits = torch.mean(torch.stack(outputs), dim=0)
#     predicted_class = int(torch.argmax(avg_logits, dim=1).item())

# print(f"Predicted class: {predicted_class}")

# LIME: Batch prediction
def batch_predict(images_np):
    batch = torch.stack([
        transform(Image.fromarray(img).convert("RGB")) for img in images_np
    ], dim=0).to(device)

    with torch.no_grad():
        outputs = [model(batch) for model in ensemble_models]
        logits = torch.mean(torch.stack(outputs), dim=0)
    return logits.cpu().numpy()

# LIME Explanation
def get_lime_explanation(pil_img, target_class):
    explainer = lime_image.LimeImageExplainer()
    img_np = np.array(pil_img)

    explanation = explainer.explain_instance(
        img_np,
        batch_predict,
        labels=(target_class,),
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    if target_class not in explanation.local_exp:
        raise ValueError(f"Target class {target_class} not found in LIME explanation.")

    lime_img, mask = explanation.get_image_and_mask(
        label=target_class,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    return lime_img, mask

# Grad-CAM Explanation (using ResNet50)
def get_gradcam_overlay(input_tensor, image_pil, target_class):
    model_for_cam = model_1
    target_layers = [model_for_cam.layer4[-1]]

    cam = GradCAM(model=model_for_cam, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]

    img_np = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

def Explanation(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = [model(input_tensor) for model in ensemble_models]
        avg_logits = torch.mean(torch.stack(outputs), dim=0)
        predicted_class = int(torch.argmax(avg_logits, dim=1).item())

    lime_img, lime_mask = get_lime_explanation(image_pil, predicted_class)
    gradcam_img = get_gradcam_overlay(input_tensor, image_pil, predicted_class)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(np.array(image_pil))
    axs[0].set_title("Original X-ray")
    axs[0].axis("off")

    axs[1].imshow(mark_boundaries(lime_img / 255.0, lime_mask))
    axs[1].set_title("LIME Explanation")
    axs[1].axis("off")

    axs[2].imshow(gradcam_img)
    axs[2].set_title("Grad-CAM Explanation (ResNet50)")
    axs[2].axis("off")

    plt.tight_layout()

    # Save figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    with torch.no_grad():
        outputs = [model(input_tensor) for model in ensemble_models]
        avg_logits = torch.mean(torch.stack(outputs), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=1)[0]
    
    # Create JSON object with all labels and their percentages
    labels_dict = {class_names[i]: round(float(probabilities[i].item()) * 100, 2) for i in range(len(class_names))}

    return buf, labels_dict

# returslt_image,label = Explanation("C:\\Users\\amirt\\OneDrive\\Desktop\\finalproject\\Lung Disease Dataset\\test\\Corona Virus Disease\\_21_3861426.jpeg")
# print(label)


