import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from sign_language import config, inference
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- Load Model and Class Names ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inference.load_trained_model(device=device)
model.eval() # Set model to evaluation mode

from sign_language.data import get_class_names
all_labels = get_class_names()

# --- MediaPipe Settings ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Transform Definition ---
preprocess = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
])

# --- Video Capture ---
cap = cv2.VideoCapture(0)

print("System started. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        all_x = []
        all_y = []
        h, w, c = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(lm.x * w)
                all_y.append(lm.y * h)
            
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate bounding box coordinates for all hands combined
        x_min = int(min(all_x)) - 10
        y_min = int(min(all_y)) - 10
        x_max = int(max(all_x)) + 10
        y_max = int(max(all_y)) + 10

        # Clipping to stay within frame boundaries
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0 and x_max > x_min and y_max > y_min:
            roi_pil = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = transforms.ToPILImage()(roi_pil)
            
            # Apply preprocessing transform
            roi_tensor = preprocess(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(roi_tensor)
                
                # Calculate probabilities
                probabilities = F.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                
                # Convert to percentage
                conf_percent = int(confidence.item() * 100)
                
                # Get predicted name with probability
                predicted_text = f"{all_labels[pred.item()]} ({conf_percent}%)"

                # Visualization
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Exit with ESC
        break

cap.release()
cv2.destroyAllWindows()
