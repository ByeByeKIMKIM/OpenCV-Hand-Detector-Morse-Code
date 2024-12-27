import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import math
import time
import speech_recognition as sr

#video capture settings (fps stuff isn't working)
capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# capture.set(cv2.CAP_PROP_FPS, 60)

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

#create a hand detector and set parameters
detector = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.9,
    )

#morsecode to english letters dictionary
morse_dict = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
        '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
        '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
        '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
        '-.--': 'Y', '--..': 'Z',
        '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
        '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
        '.-.-.-': '.', '--..--': ',', '..--..': '?', '-.-.--': '!', '-....-': '-',
        '-..-.': '/', '.--.-.': '@', '-.--.': '(', '-.--.-': ')'
    }

#morsecode to english translator function
def morseToEnglish(morseCode):
    morseCodeSplit = morseCode.split()
    translation = ""
    try:
        for code in morseCodeSplit:
            translation += morse_dict[code]
        return translation
    except KeyError:
        return ""

#main tracking function
def run_tracking():
    pTime = 0

    detecting = False

    thumb_in_state = False
    pointer_thumb_state = False
    middle_in_state = False

    #continually run
    morse = ""
    translation = ""
    while True:
        isFist = False
        pointer_and_thumb = False
        pointer_and_middle = False
        thumb_in = False
        middle_in = False

        #get the frame
        success, frame = capture.read()
        #flip the frame so it mirrors and looks more correct
        flippedCapture = cv2.flip(frame, 1)

        #add all the other stuff after to make sure it doesn't get flipped

        #openCV captures images in BGR, but mediapipe requires RGB
        frame_rgb = cv2.cvtColor(flippedCapture, cv2.COLOR_BGR2RGB)

        #results of tracking hand landmarks
        results = detector.process(frame_rgb)

        #if hand landmarks were able to be tracked
        if results.multi_hand_landmarks:
            #for each landmark
            for hand_landmarks in results.multi_hand_landmarks:
                #draw the landmark
                mp_drawing.draw_landmarks(
                    image=flippedCapture,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )
                h, w, _ = flippedCapture.shape

                mark_9 = hand_landmarks.landmark[9]
                mark_0 = hand_landmarks.landmark[0]
                mark_6 = hand_landmarks.landmark[6]

                #getting the center of the palm
                mark_9_x, mark_9_y = int(mark_9.x * w), int(mark_9.y * h)
                mark_0_x, mark_0_y = int(mark_0.x * w), int(mark_0.y * h)
                mark_9_0_middle_x = int(abs((mark_9_x + mark_0_x) / 2))
                mark_9_0_middle_y = int(abs((mark_9_y + mark_0_y) / 2))
                #cv2.circle(flippedCapture, (mark_9_0_middle_x, mark_9_0_middle_y), 10, (0, 0, 225), -1)

                #drawing a dircle with center in the middle of the palm, and out to landmark 6 (this will be used to detect a fist)
                dist_mark_6_center = int(math.sqrt((mark_6.x * w - mark_9_0_middle_x) ** 2 + (mark_6.y * h - mark_9_0_middle_y) ** 2))
                optimal_dist_mark_6_center = dist_mark_6_center + 50
                #cv2.circle(flippedCapture, (mark_9_0_middle_x, mark_9_0_middle_y), optimal_dist_mark_6_center, (0, 0, 225), 10)

                #checking if a fist has been made
                is_fist_temp = True
                for landmark in hand_landmarks.landmark:
                    landmark_dist_from_center = int(math.sqrt((landmark.x * w - mark_9_0_middle_x) ** 2 + (landmark.y * h - mark_9_0_middle_y) ** 2))
                    if landmark_dist_from_center > optimal_dist_mark_6_center:
                        is_fist_temp = False;
                isFist = is_fist_temp

                #checking if pointer finger and thumb are touching
                mark_8 = hand_landmarks.landmark[8]
                mark_4 = hand_landmarks.landmark[4]
                mark_7 = hand_landmarks.landmark[7]
                mark_3 = hand_landmarks.landmark[3]
                mark_8_x, mark_8_y = int(mark_8.x * w), int(mark_8.y * h)
                mark_4_x, mark_4_y = int(mark_4.x * w), int(mark_4.y * h)
                #cv2.circle(flippedCapture, (mark_8_x, mark_8_y), int(math.sqrt((mark_8_x - mark_7.x * w) ** 2 + (mark_8_y - mark_7.y * h) ** 2)), (0, 225, 225), 2)
                #cv2.circle(flippedCapture, (mark_4_x, mark_4_y), int(math.sqrt((mark_4_x - mark_3.x * w) ** 2 + (mark_4_y - mark_3.y * h) ** 2)), (0, 225, 225), 2)
                if int(math.sqrt((mark_8_x - mark_4_x) ** 2 + (mark_8_y - mark_4_y) ** 2)) < int(math.sqrt((mark_4_x - mark_3.x * w) ** 2 + (mark_4_y - mark_3.y * h) ** 2)):
                    pointer_and_thumb = True

                #checking if pointer and middle are touching
                mark_12 = hand_landmarks.landmark[12]
                #cv2.circle(flippedCapture, (mark_8_x, mark_8_y), int(math.sqrt((mark_8_x - mark_7.x * w) ** 2 + (mark_8_y - mark_7.y * h) ** 2)), (0, 225, 225), 2)
                if int(math.sqrt((mark_8_x - mark_12.x * w) ** 2 + (mark_8_y - mark_12.y * h) ** 2)) < int(math.sqrt((mark_8_x - mark_7.x * w) ** 2 + (mark_8_y - mark_7.y * h) ** 2)) + 40:
                    pointer_and_middle = True

                #checking if thumb has been moved in
                mark_10 = hand_landmarks.landmark[10]
                mark_5 = hand_landmarks.landmark[5]
                dist_thumb_from_center = math.sqrt((mark_4_x-mark_9_0_middle_x) ** 2 + (mark_4_y - mark_9_0_middle_y) ** 2)
                if dist_thumb_from_center < math.sqrt((mark_5.x * w - mark_9_0_middle_x) ** 2 + (mark_5.y * h - mark_9_0_middle_y) ** 2):
                    thumb_in = True

                #checking if middle has been moved in
                dist_middle_from_center = math.sqrt((mark_12.x * w - mark_9_0_middle_x) ** 2 + (mark_12.y * h - mark_9_0_middle_y) ** 2)
                if dist_middle_from_center < math.sqrt((mark_5.x * w - mark_9_0_middle_x) ** 2 + (mark_5.y * h - mark_9_0_middle_y) ** 2):
                    middle_in = True

                if thumb_in and not thumb_in_state:
                    morse += "."
                    thumb_in_state = True
                elif not thumb_in:
                    thumb_in_state = False

                if pointer_and_thumb and not pointer_thumb_state:
                    morse += "-"
                    pointer_thumb_state = True
                elif not pointer_and_thumb:
                    pointer_thumb_state = False

                if middle_in and not middle_in_state:
                    morse += " "
                    middle_in_state = True
                elif not middle_in:
                    middle_in_state = False

                if middle_in and thumb_in:
                    morse = ""

                translation = morseToEnglish(morse)
        #cv2.putText(flippedCapture, "Fist" if isFist else "Not Fist", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(flippedCapture, "Pointer Thumb Touch" if pointer_and_thumb else "Not Pointer Thumb Touch", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(flippedCapture, "Pointer Middle Touch" if pointer_and_middle else "Not Pointer Middle Touch", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(flippedCapture, "Thumb In" if thumb_in else "Not Thumb In", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(flippedCapture, "Middle In" if middle_in else "Not Middle In", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(flippedCapture, morse, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(flippedCapture, translation, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        #cv2.putText(flippedCapture, "Middle In" if middle_in else "Not Middle In", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #fps stuff
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        #cv2.putText(flippedCapture, f'FPS: {int(fps)}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        #show the capture
        cv2.imshow("Tracking", flippedCapture)
        cv2.waitKey(1)

#main method
if __name__ == '__main__':
    run_tracking()