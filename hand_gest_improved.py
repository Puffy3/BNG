import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, detection_confidence=0.5):
        self.detection_confidence = detection_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                          min_detection_confidence=self.detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for lm_id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([lm_id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def count_fingers(self, lm_list):
        # Define landmark indices for thumb, index, middle, ring, and little fingers
        tip_ids = [4, 8, 12, 16, 20]

        fingers = []

        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[0][1]:  # Right hand
            fingers.append(lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1])
        else:  # Left hand
            fingers.append(lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1])

        # Other fingers (index, middle, ring, little)
        for id in range(1, 5):
            fingers.append(lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2])

        return fingers.count(True)  # Count how many fingers are extended

# Example usage:
detector = HandDetector(detection_confidence=0.75)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if lm_list:
        fingers = detector.count_fingers(lm_list)
        print("Number of fingers:", fingers)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

