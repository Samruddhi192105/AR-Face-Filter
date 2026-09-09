import cv2
import mediapipe as mp
import math


# -----------------------------
# MediaPipe Face Landmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(options)


# -----------------------------
# Load Hat
# -----------------------------

hat = cv2.imread(
    "assets/hat.png",
    cv2.IMREAD_UNCHANGED
)

if hat is None:
    print("Could not load hat.png")
    exit()

if hat.shape[2] != 4:
    print("hat.png must have a transparent background")
    exit()


# -----------------------------
# Rotate Image
# -----------------------------

def rotate_image(image, angle):

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face landmarks
    result = landmarker.detect(mp_image)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # -----------------------------
        # Landmark points
        # -----------------------------

        left_point = landmarks[234]
        right_point = landmarks[454]
        forehead_point = landmarks[10]

        left_x = int(left_point.x * width)
        left_y = int(left_point.y * height)

        right_x = int(right_point.x * width)
        right_y = int(right_point.y * height)

        forehead_x = int(
            forehead_point.x * width
        )

        forehead_y = int(
            forehead_point.y * height
        )

        # -----------------------------
        # Calculate face width
        # -----------------------------

        dx = right_x - left_x
        dy = right_y - left_y

        face_width = math.sqrt(
            dx ** 2 + dy ** 2
        )

        # -----------------------------
        # Calculate head tilt
        # -----------------------------

        angle = -math.degrees(
            math.atan2(dy, dx)
        )

        # -----------------------------
        # Hat size
        # -----------------------------

        hat_width = int(
            face_width * 2.0
        )

        hat_height = int(
            hat.shape[0]
            * hat_width
            / hat.shape[1]
        )

        resized_hat = cv2.resize(
            hat,
            (hat_width, hat_height)
        )

        # -----------------------------
        # Rotate hat
        # -----------------------------

        rotated_hat = rotate_image(
            resized_hat,
            angle
        )

        rotated_height, rotated_width = (
            rotated_hat.shape[:2]
        )

        # -----------------------------
        # Hat position
        # -----------------------------

        x1 = (
            forehead_x
            - rotated_width // 2
        )

        y1 = (
            forehead_y
            - rotated_height
            + int(face_width * 0.25)
        )

        x2 = x1 + rotated_width
        y2 = y1 + rotated_height

        # -----------------------------
        # Boundary check
        # -----------------------------

        if (
            x1 >= 0
            and y1 >= 0
            and x2 <= width
            and y2 <= height
        ):

            roi = frame[
                y1:y2,
                x1:x2
            ]

            # -----------------------------
            # Alpha blending
            # -----------------------------

            alpha = (
                rotated_hat[:, :, 3]
                / 255.0
            )

            alpha = alpha[:, :, None]

            foreground = (
                rotated_hat[:, :, :3]
                .astype(float)
            )

            background = (
                roi.astype(float)
            )

            blended = (
                alpha * foreground
                + (1 - alpha) * background
            )

            frame[
                y1:y2,
                x1:x2
            ] = blended.astype("uint8")

    # -----------------------------
    # Display
    # -----------------------------

    cv2.imshow(
        "AR Hat Filter",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()