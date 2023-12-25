import cv2

if __name__ == '__main__':

    # Load the face and eye cascade files
    # https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/anaustinbeing/haar-cascade-files/blob/master/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        check, frame = cap.read()

        # Make a copy of the frame
        frame_processed = frame.copy()
        # Conver to GrayScale (Haar Cascade algorithm works on gray scale images)
        gray_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)

        # Run the face detection algorithm
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            frame_processed = cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_frame[y:y + h, x:x + w]
            roi_color = frame_processed[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Show frame
        cv2.imshow('video stream', frame_processed)

        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
