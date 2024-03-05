import cv2
import mediapipe as mp
import pyautogui

# Function to move the mouse cursor based on hand position with reduced speed
def move_mouse(x, y, width, height, speed_factor=1):
    screen_width, screen_height = pyautogui.size()
    move_x = int(x * screen_width / width)
    move_y = int(y * screen_height / height)
    current_x, current_y = pyautogui.position()
    new_x = current_x + (move_x - current_x) / speed_factor
    new_y = current_y + (move_y - current_y) / speed_factor
    pyautogui.moveTo(new_x, new_y)

# Main function for hand tracking and mouse control
def main():
    cap = cv2.VideoCapture(0)

    # Initialize Mediapipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark coordinates
                landmarks = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]

                # Draw landmarks on the frame
                for landmark in landmarks:
                    x, y = landmark
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                # Calculate distances between finger tips
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]

                # Calculate distances between fingers
                thumb_index_dist = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
                index_middle_dist = ((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)**0.5
                thumb_palm_dist = ((thumb_tip[0] - landmarks[0][0])**2 + (thumb_tip[1] - landmarks[0][1])**2)**0.5

                # Perform actions based on finger positions
                if thumb_index_dist < 50:
                    pyautogui.click()
                elif thumb_palm_dist < 50:
                    pyautogui.doubleClick()
                elif thumb_palm_dist > 150:
                    move_mouse(sum(x for x, _ in landmarks) / len(landmarks), sum(y for _, y in landmarks) / len(landmarks), width, height)
                elif index_middle_dist < 30 and middle_tip[1] < landmarks[0][1]:
                    pyautogui.mouseDown()
                elif index_middle_dist < 30 and middle_tip[1] > landmarks[0][1]:
                    pyautogui.mouseUp()

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
