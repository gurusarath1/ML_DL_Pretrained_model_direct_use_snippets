import cv2
import mediapipe as mp

if __name__ == '__main__':

    print('Running Pose Estimation ...')

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # VIDEO
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Video Running..', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    print('Done')
