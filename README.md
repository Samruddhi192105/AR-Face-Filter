# 🕶️ AR Face Filter

A real-time **Augmented Reality Face Filter** built using **Python, OpenCV, and MediaPipe Face Landmarker**.

The application uses facial landmarks detected from a webcam to dynamically place and track **sunglasses, a mustache, and a hat** on the user's face.

The overlays automatically adjust their **position, size, and rotation** according to the user's facial movements and head tilt.

---

## 🎯 Project Objective

The goal of this project is to understand and implement the fundamental concepts behind real-time AR filters:

* Image processing
* Face detection
* Facial landmark detection
* Coordinate geometry
* Distance-based scaling
* Head-tilt estimation
* Image rotation
* Alpha blending
* Real-time webcam processing

---

## ✨ Features

### 🕶️ Sunglasses Overlay

* Uses eye landmarks to locate the eyes.
* Calculates the distance between the eyes.
* Dynamically scales the sunglasses according to face size.
* Positions the sunglasses around the eyes.
* Rotates the sunglasses according to head tilt.

### 👨 Mustache Overlay

* Uses mouth landmarks to locate the mouth.
* Calculates mouth width.
* Dynamically scales the mustache.
* Positions the mustache below the nose/mouth area.
* Rotates the mustache according to head tilt.

### 🎩 Hat Overlay

* Uses facial landmarks around the sides and forehead.
* Estimates face width.
* Dynamically scales the hat.
* Positions the hat above the forehead.
* Rotates the hat according to head tilt.

### 🎥 Real-Time Tracking

The system processes webcam frames continuously and updates the AR overlays according to the user's movements.

### ⚡ FPS Monitoring

The application displays the current frames per second to monitor real-time performance.

---

## 🧠 How It Works

```text
                 Webcam
                    ↓
              Video Frame
                    ↓
          BGR → RGB Conversion
                    ↓
        MediaPipe Face Landmarker
                    ↓
          Facial Landmark Points
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Eyes        Mouth       Head/Face
        ↓           ↓           ↓
   Sunglasses    Mustache       Hat
        ↓           ↓           ↓
        └───────────┼───────────┘
                    ↓
              Alpha Blending
                    ↓
              Final AR Frame
                    ↓
                 Display
```

---

## 🛠️ Tech Stack

| Technology | Purpose                              |
| ---------- | ------------------------------------ |
| Python     | Programming language                 |
| OpenCV     | Image processing and webcam handling |
| MediaPipe  | Facial landmark detection            |
| NumPy      | Numerical/image operations           |
| Git        | Version control                      |
| GitHub     | Source code hosting                  |

---

## 📂 Project Structure

```text
AR-Face-Filter/
│
├── assets/
│   ├── sunglasses.png
│   ├── mustache.png
│   └── hat.png
│
├── models/
│   └── face_landmarker.task
│
├── image_basics.py
├── face_detection.py
├── face_landmarks.py
├── sunglasses_overlay.py
├── mustache_overlay.py
├── hat_overlay.py
├── final_ar_filter.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Samruddhi192105/AR-Face-Filter.git
```

Move into the project directory:

```bash
cd AR-Face-Filter
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Git Bash:

```bash
source venv/Scripts/activate
```

For PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the final AR filter:

```bash
python final_ar_filter.py
```

A webcam window will open.

The application will detect your face and apply:

```text
🎩 Hat
🕶️ Sunglasses
👨 Mustache
```

Press:

```text
Q
```

to exit the application.

---
