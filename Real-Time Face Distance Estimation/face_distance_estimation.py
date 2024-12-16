import cv2


def calculate_distance(face_width, focal_len, avg_height):
    return (focal_len * avg_height) / face_width


capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

avg_face_height = 15.0
focal_length = 1500.0

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        distance = calculate_distance(w, focal_length, avg_face_height)

        cv2.putText(frame, f"Distance: {int(distance)} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-time Face Distance Measurement', frame)

    if cv2.waitKey(10) == 0x1b:
        print('ESC pressed. Exiting...')
        break

capture.release()
cv2.destroyAllWindows()
