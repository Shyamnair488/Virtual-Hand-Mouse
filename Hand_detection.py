import cv2
import mediapipe as mp
import pyautogui

# Function to move the mouse cursor based on hand position
def move_mouse(x, y, width, height):
    # Normalize hand position to screen resolution
    screen_width, screen_height = pyautogui.size()
    move_x = int(x * screen_width / width)
    move_y = int(y * screen_height / height)
    pyautogui.moveTo(move_x, move_y)

# Main function for hand tracking and mouse control
def main():
    cap = cv2.VideoCapture(0)

    # Initialize Mediapipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe hands
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Get the centroid of the hand (average of all landmarks)
                cx = int(sum([landmark.x for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark) * width)
                cy = int(sum([landmark.y for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark) * height)
                # Move the mouse cursor
                move_mouse(cx, cy, width, height)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
