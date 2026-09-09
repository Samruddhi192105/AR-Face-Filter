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
# Create Face Landmarker
# --------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    print("Face landmark tracking started")
    print("Press Q to quit")

    while True:

        # --------------------------------
        # Read webcam frame
        # --------------------------------

        success, frame = cap.read()

        if not success:
            print("Failed to read frame")
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

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # --------------------------------
        # Detect face landmarks
        # --------------------------------

        result = landmarker.detect(mp_image)

        if result.face_landmarks:

            # Get first detected face
            face_landmarks = result.face_landmarks[0]

            # --------------------------------
            # Eye landmark indices
            # --------------------------------

            LEFT_EYE_1 = 33
            LEFT_EYE_2 = 133

            RIGHT_EYE_1 = 362
            RIGHT_EYE_2 = 263

            # --------------------------------
            # Convert normalized coordinates
            # to pixel coordinates
            # --------------------------------

            left_eye_1 = face_landmarks[LEFT_EYE_1]
            left_eye_2 = face_landmarks[LEFT_EYE_2]

            right_eye_1 = face_landmarks[RIGHT_EYE_1]
            right_eye_2 = face_landmarks[RIGHT_EYE_2]

            left_p1 = (
                int(left_eye_1.x * width),
                int(left_eye_1.y * height)
            )

            left_p2 = (
                int(left_eye_2.x * width),
                int(left_eye_2.y * height)
            )

            right_p1 = (
                int(right_eye_1.x * width),
                int(right_eye_1.y * height)
            )

            right_p2 = (
                int(right_eye_2.x * width),
                int(right_eye_2.y * height)
            )

            # --------------------------------
            # Calculate eye centers
            # --------------------------------

            left_center = (
                (left_p1[0] + left_p2[0]) // 2,
                (left_p1[1] + left_p2[1]) // 2
            )

            right_center = (
                (right_p1[0] + right_p2[0]) // 2,
                (right_p1[1] + right_p2[1]) // 2
            )

            # --------------------------------
            # Draw eye centers
            # --------------------------------

            cv2.circle(
                frame,
                left_center,
                5,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                right_center,
                5,
                (0, 0, 255),
                -1
            )

            # --------------------------------
            # Calculate eye distance
            # --------------------------------

            dx = right_center[0] - left_center[0]
            dy = right_center[1] - left_center[1]

            eye_distance = math.sqrt(
                dx ** 2 + dy ** 2
            )

            # --------------------------------
            # Calculate eye angle
            # --------------------------------

            angle = math.degrees(
                math.atan2(dy, dx)
            )

            # --------------------------------
            # Calculate center between eyes
            # --------------------------------

            center_x = (
                left_center[0] + right_center[0]
            ) // 2

            center_y = (
                left_center[1] + right_center[1]
            ) // 2

            eye_center = (
                center_x,
                center_y
            )

            # --------------------------------
            # Draw line between eyes
            # --------------------------------

            cv2.line(
                frame,
                left_center,
                right_center,
                (255, 0, 0),
                2
            )

            # Draw center between eyes
            cv2.circle(
                frame,
                eye_center,
                6,
                (0, 255, 255),
                -1
            )

            # --------------------------------
            # Display information
            # --------------------------------

            cv2.putText(
                frame,
                f"Eye Distance: {eye_distance:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Eye Angle: {angle:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Center: {eye_center}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Print values in terminal
            print(
                f"Distance: {eye_distance:.2f} | "
                f"Angle: {angle:.2f} | "
                f"Center: {eye_center}"
            )

        # --------------------------------
        # Display webcam
        # --------------------------------

        cv2.imshow(
            "Eye Landmark Tracking",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------
    # Cleanup
    # --------------------------------

    cap.release()
    cv2.destroyAllWindows()