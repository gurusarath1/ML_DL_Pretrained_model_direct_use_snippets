import cv2
import mediapipe as mp

if __name__ == '__main__':

    print('Running Pose Estimation ...')

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles

    # VIDEO
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as faces:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = faces.process(image)

            if not results.multi_face_landmarks:
                continue

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.multi_face_landmarks[0],
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
                                      )

            cv2.imshow('Video Running..', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    print('Done')
