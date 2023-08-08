import cv2

if __name__ == '__main__':

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW))
    while cap.isOpened():
        check, frame = cap.read()

        cv2.imshow('video stream', frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
