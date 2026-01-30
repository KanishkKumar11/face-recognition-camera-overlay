# Face Recognition Camera Overlay (OpenCV)

This project uses **OpenCV** to overlay a live webcam feed onto a static background image
with controlled size, aspect ratio, and positioning — ensuring it does not overlap
key background elements.

## 🎯 Features
- Live webcam capture using OpenCV
- Tall & narrow camera frame customization
- Fixed left-side placement to avoid background overlap
- Adjustable vertical offset
- Real-time rendering

## 🛠 Tech Stack
- Python
- OpenCV
- NumPy

## 📂 Project Structure
.
├── main.py
├── Resources/
│ └── IMG_0300.jpeg
├── Images/
│ └── sample images
└── README.md


## ▶️ How to Run
```bash
pip install opencv-python numpy
python main.py