import torch
from torchvision import transforms
from src.models.model import build_model
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
fig.patch.set_visible(False)
fig1.patch.set_visible(False)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig.patch.set_visible(False)
fig1.patch.set_visible(False)

ax.axis("off")
ax1.axis("off")



device = torch.device("cpu")

mtcnn = MTCNN(image_size = 160, margin = 20, min_face_size = 40, thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True, device = device)

frameCounter = 10

current_label = "loading"
last_emotion = None


device = torch.device("cpu")

mtcnn = MTCNN(image_size = 160, margin = 20, min_face_size = 40, thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True, device = device)

# 1. Bildpfad
#imagepath = "D:/Uni/Semester III/Praktikum/Projekt/3/CV-DL-FER-Project-/src/datasets/FER2013/test/happy/PrivateTest_556985.jpg"

# 2. Modell laden
checkpoint = torch.load("C:/Users/Vinzenz/checkpoint_epoch_61.pt", map_location='cpu')

# Versuche num_classes aus dem Checkpoint zu bestimmen (falls vorhanden)
num_classes = None
if isinstance(checkpoint, dict):
    num_classes = checkpoint.get('num_classes')

# Fallback: falls num_classes nicht im Checkpoint steht, try to infer from fc.weight in the state_dict
state_dict = None
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model')
else:
    state_dict = checkpoint

if num_classes is None and isinstance(state_dict, dict):
    for k, v in state_dict.items():
        if k.endswith('fc.weight'):
            try:
                num_classes = v.shape[0]
                print(f"Detected num_classes={num_classes} from checkpoint key '{k}'")
            except Exception:
                pass
            break

# Letzte Fallback
if num_classes is None:
    num_classes = 7  # default

# Build model with the detected/assumed number of classes
model = build_model("resnet18", num_classes=num_classes, input_channels=1, small_input=True)

# Versuche das state dict zu laden. Wenn Dimensionen der finalen Schicht nicht passen, lade alle anderen Gewichte.
if state_dict is not None:
    try:
        model.load_state_dict(state_dict)
        print("Model weights loaded from checkpoint.")
    except RuntimeError as e:
        print("Direct load failed:", e)
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace('module.', '')] = v

        # Prüfe ob fc Gewicht/Bias in den Checkpoint Keys sind und ob ihre Shapes passen
        fc_keys = ['fc.weight', 'fc.bias']
        mismatch = False
        for key in fc_keys:
            if key in new_state and key in dict(model.named_parameters()):
                if new_state[key].shape != dict(model.named_parameters())[key].shape:
                    print(f"Shape mismatch for {key}: checkpoint {new_state[key].shape} vs model {dict(model.named_parameters())[key].shape}")
                    mismatch = True

        if mismatch:
            # Entferne finale Schicht aus dem geladenen state_dict und lade den Rest
            for key in fc_keys:
                if key in new_state:
                    print(f"Removing {key} from checkpoint before loading")
                    del new_state[key]
            model.load_state_dict(new_state, strict=False)
            print("Model weights loaded with final layer skipped (will be randomly initialized).")
        else:
            # Falls kein Shape-Mismatch, versuche zu laden
            model.load_state_dict(new_state)
            print("Model weights loaded after stripping 'module.' prefix.")

model.to(device)
model.eval()

# labels aus checkpoint falls vorhanden
if isinstance(checkpoint, dict) and checkpoint.get('class_names'):
    labels = checkpoint.get('class_names')
else:
    labels = ["angry","disgust","fear","happy","sad","surprise"]


def determineEmotion():
    # 4. Verwende das zuletzt erkannte 64x64-Gesicht (als numpy-array `face64`)
    global face64
    try:
        img = Image.fromarray(face64).convert("L")
    except Exception:
        print("Kein gültiges Gesichtspatch für Inferenz vorhanden.")
        return None

    # lokale Transform-Pipeline: 64x64, Graustufen, Normalisierung
    transform_local = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_tensor = transform_local(img).unsqueeze(0).float()  # (1, C, H, W)

    # 5. Inference
    if img_tensor is None:
        print("Kein Gesicht im Bild erkannt. Überprüfe das Bild oder MTCNN-Parameter.")
        return None
    else:
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = int(probs.argmax())
            predicted_label = labels[predicted_class] if labels and len(labels) > predicted_class else str(predicted_class)
        #print(f"Predicted class: {predicted_class} ({predicted_label})")
        #print full probabilities per label
        #print("Probabilities:")
        for i, p in enumerate(probs):
            name = labels[i] if i < len(labels) else str(i)
            #print(f"  {i} ({name}): {p:.4f}")
        #print(probs)
        return (probs.argmax())
    

def processImage(image):
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        # kein Gesicht → neutrales Bild + keine Box
        return np.full((64, 64), 127, dtype=np.uint8), None

    # bestes Gesicht wählen
    idx = int(np.argmax(probs)) if probs is not None else 0
    x1, y1, x2, y2 = map(int, boxes[idx])

    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    # Gesicht extrahieren
    face = image.crop((x1, y1, x2, y2)).resize((64, 64)).convert("L")
    face64 = np.array(face)

    return face64, (x1, y1, x2, y2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    ax.clear()
    ax.imshow(img)

    
    ax1.clear()
    face64,box = processImage(img)
    if box is not None:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
        ax.add_patch(rect)
        # show last predicted emotion (updated every 30 frames)
        if last_emotion is not None:
            ax.text(x1, y2 + 50, labels[last_emotion], color="lime")


    ax.axis("off")          # Achsen aus
    ax.set_frame_on(False)  # Rahmen aus

    ax1.axis("off")          # Achsen aus
    ax1.set_frame_on(False)  # Rahmen aus

    ax1.imshow(face64, cmap='gray', vmin=0, vmax=255)
    if frameCounter < 10:
        frameCounter += 1
    else:
        frameCounter = 0
        # Only run emotion detection when a face is present
        if 'box' in locals() and box is not None:
            last_emotion = determineEmotion()
            print("Updated emotion:", labels[last_emotion])
        else:
            print("No face present when trying to determine emotion; skipping.")

    #print (frameCounter)

    plt.pause(0.001)

cap.release()