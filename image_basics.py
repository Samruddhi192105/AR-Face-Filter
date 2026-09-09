import cv2

# -----------------------------
# 1. Read image
# -----------------------------

image = cv2.imread("assets/plant.png")

if image is None:
    print("Image not found")
    exit()

# -----------------------------
# 2. Image dimensions
# -----------------------------

height, width, channels = image.shape

print("Height:", height)
print("Width:", width)
print("Channels:", channels)

# -----------------------------
# 3. Grayscale
# -----------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 4. HSV
# -----------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# -----------------------------
# 5. Resize
# -----------------------------

resized = cv2.resize(image, (400, 300))

# -----------------------------
# 6. Crop
# -----------------------------

cropped = image[100:400, 150:500]

# -----------------------------
# 7. Rotate
# -----------------------------

center = (width // 2, height // 2)

M = cv2.getRotationMatrix2D(
    center,
    30,
    1.0
)

rotated = cv2.warpAffine(
    image,
    M,
    (width, height)
)

# -----------------------------
# Display images at smaller size
# -----------------------------

display_width = 600
display_height = 400

display_original = cv2.resize(
    image,
    (display_width, display_height)
)

display_gray = cv2.resize(
    gray,
    (display_width, display_height)
)

display_hsv = cv2.resize(
    hsv,
    (display_width, display_height)
)

display_cropped = cv2.resize(
    cropped,
    (display_width, display_height)
)

display_rotated = cv2.resize(
    rotated,
    (display_width, display_height)
)

cv2.imshow("Original", display_original)
cv2.imshow("Grayscale", display_gray)
cv2.imshow("HSV", display_hsv)
cv2.imshow("Cropped", display_cropped)
cv2.imshow("Rotated", display_rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()