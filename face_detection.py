import cv2

# ---------------------------------
# 1. Load Haar Cascade
# ---------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Failed to load Haar Cascade")
    exit()

print("Haar Cascade loaded successfully")


# ---------------------------------
# 2. Open webcam
# ---------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam")
    exit()


# ---------------------------------
# 3. Main camera loop
# ---------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # ---------------------------------
    # 4. Convert BGR to grayscale
    # ---------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------
    # 5. Detect faces
    # ---------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ---------------------------------
    # 6. Draw rectangles
    # ---------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ---------------------------------
    # 7. Display result
    # ---------------------------------

    cv2.imshow(
        "Face Detection",
        frame
    )

    # ---------------------------------
    # 8. Quit
    # ---------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------
# 9. Release resources
# ---------------------------------

cap.release()
cv2.destroyAllWindows()