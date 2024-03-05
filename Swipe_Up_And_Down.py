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
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_hand_landmarks = None
        right_hand_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark coordinates
                landmarks = [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]

                # Determine left or right hand based on position
                if landmarks[0][0] < width / 2:  # Left hand
                    left_hand_landmarks = landmarks
                    color = (0, 255, 255)  # Yellow color for left hand landmarks
                else:  # Right hand
                    right_hand_landmarks = landmarks
                    color = (0, 255, 0)  # Green color for right hand landmarks

                # Draw landmarks
                for landmark in landmarks:
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 5, color, -1)

            # Perform actions with the right hand
            if right_hand_landmarks:
                # Calculate distances between finger tips
                thumb_tip = right_hand_landmarks[4]
                index_tip = right_hand_landmarks[8]
                middle_tip = right_hand_landmarks[12]
                ring_tip = right_hand_landmarks[16]
                pinky_tip = right_hand_landmarks[20]

                # Calculate distances between fingers
                thumb_index_dist = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
                index_middle_dist = ((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)**0.5
                thumb_palm_dist = ((thumb_tip[0] - right_hand_landmarks[0][0])**2 + (thumb_tip[1] - right_hand_landmarks[0][1])**2)**0.5

                # Perform actions based on finger positions
                if thumb_index_dist < 50:
                    pyautogui.click()
                elif thumb_palm_dist < 50:
                    pyautogui.doubleClick()
                elif thumb_palm_dist > 150:
                    move_mouse(sum(x for x, _ in right_hand_landmarks) / len(right_hand_landmarks), sum(y for _, y in right_hand_landmarks) / len(right_hand_landmarks), width, height)
                elif index_middle_dist < 30 and middle_tip[1] < right_hand_landmarks[0][1]:
                    pyautogui.mouseDown()
                elif index_middle_dist < 30 and middle_tip[1] > right_hand_landmarks[0][1]:
                    pyautogui.mouseUp()

            # Perform scrolling with the left hand
            if left_hand_landmarks:
                # Calculate vertical distance between thumb and index finger
                thumb_y = left_hand_landmarks[4][1]
                index_y = left_hand_landmarks[8][1]
                thumb_index_dist_y = abs(thumb_y - index_y)

                # Scroll up or down based on finger positions
                if thumb_index_dist_y > 50:
                    if thumb_y < index_y:  # Scroll down
                        pyautogui.scroll(-100)
                    else:  # Scroll up
                        pyautogui.scroll(100)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
