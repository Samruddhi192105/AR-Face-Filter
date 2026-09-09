import cv2
import mediapipe as mp
import math


# --------------------------------
# MediaPipe Face Landmarker
# --------------------------------

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


# --------------------------------
# Load mustache
# --------------------------------

mustache = cv2.imread(
    "assets/mustache.png",
    cv2.IMREAD_UNCHANGED
)

if mustache is None:
    print("Error: mustache.png not found")
    exit()

print("Mustache loaded")
print("Mustache shape:", mustache.shape)


# --------------------------------
# Rotate image
# --------------------------------

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


# --------------------------------
# Create Face Landmarker
# --------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    print("AR mustache started")
    print("Press Q to quit")

    while True:

        # --------------------------------
        # Read frame
        # --------------------------------

        success, frame = cap.read()

        if not success:
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # --------------------------------
        # BGR → RGB
        # --------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # --------------------------------
        # Detect landmarks
        # --------------------------------

        result = landmarker.detect(mp_image)

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            # --------------------------------
            # Important landmarks
            # --------------------------------

            NOSE = 1
            MOUTH_LEFT = 61
            MOUTH_RIGHT = 291

            nose = face_landmarks[NOSE]
            mouth_left = face_landmarks[MOUTH_LEFT]
            mouth_right = face_landmarks[MOUTH_RIGHT]

            # --------------------------------
            # Convert normalized coordinates
            # to pixels
            # --------------------------------

            nose_point = (
                int(nose.x * width),
                int(nose.y * height)
            )

            mouth_left_point = (
                int(mouth_left.x * width),
                int(mouth_left.y * height)
            )

            mouth_right_point = (
                int(mouth_right.x * width),
                int(mouth_right.y * height)
            )

            # --------------------------------
            # Calculate mouth geometry
            # --------------------------------

            dx = (
                mouth_right_point[0]
                -
                mouth_left_point[0]
            )

            dy = (
                mouth_right_point[1]
                -
                mouth_left_point[1]
            )

            mouth_width = math.sqrt(
                dx ** 2 + dy ** 2
            )

            # --------------------------------
            # Calculate tilt angle
            # --------------------------------
            # Negative because webcam is mirrored

            angle = -math.degrees(
                math.atan2(dy, dx)
            )

            # --------------------------------
            # Mustache size
            # --------------------------------

            mustache_width = int(
                mouth_width * 1.8
            )

            mustache_height = int(
                mustache.shape[0]
                *
                mustache_width
                /
                mustache.shape[1]
            )

            resized = cv2.resize(
                mustache,
                (
                    mustache_width,
                    mustache_height
                ),
                interpolation=cv2.INTER_AREA
            )

            # --------------------------------
            # Rotate mustache
            # --------------------------------

            rotated = rotate_image(
                resized,
                angle
            )

            # --------------------------------
            # Get rotated dimensions
            # --------------------------------

            rotated_height, rotated_width = rotated.shape[:2]

            # --------------------------------
            # Calculate mustache position
            # --------------------------------

            center_x = (
                mouth_left_point[0]
                +
                mouth_right_point[0]
            ) // 2

            # Position between nose and mouth
            center_y = (
                nose_point[1]
                +
                mouth_left_point[1]
            ) // 2

            # Move slightly upward
            center_y -= int(
                mouth_width * 0.05
            )

            # --------------------------------
            # Position
            # --------------------------------

            x1 = int(
                center_x
                -
                rotated_width / 2
            )

            y1 = int(
                center_y
                -
                rotated_height / 2
            )

            x2 = x1 + rotated_width
            y2 = y1 + rotated_height

            # --------------------------------
            # Boundary check
            # --------------------------------

            if (
                x1 >= 0
                and y1 >= 0
                and x2 <= width
                and y2 <= height
            ):

                # --------------------------------
                # Alpha blending
                # --------------------------------

                if rotated.shape[2] == 4:

                    alpha = (
                        rotated[:, :, 3]
                        / 255.0
                    )

                    for c in range(3):

                        frame[
                            y1:y2,
                            x1:x2,
                            c
                        ] = (
                            alpha
                            *
                            rotated[:, :, c]
                            +
                            (1 - alpha)
                            *
                            frame[
                                y1:y2,
                                x1:x2,
                                c
                            ]
                        )

                else:

                    frame[
                        y1:y2,
                        x1:x2
                    ] = rotated

        # --------------------------------
        # Display
        # --------------------------------

        cv2.imshow(
            "AR Mustache",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
