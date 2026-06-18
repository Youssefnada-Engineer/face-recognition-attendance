# Face Recognition Attendance System

An AI-powered attendance system that recognizes faces and marks attendance automatically.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.100-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red)

---

## Overview

Traditional attendance systems require manual entry or ID cards. This system uses deep learning to recognize faces automatically and mark attendance in real time — no manual input needed.

---

## Features

- Recognizes faces from uploaded photos or live camera
- Marks attendance instantly with name, date, and time
- Prevents duplicate entries for the same person on the same day
- Filters and exports attendance records as CSV

---

## How It Works

1. Register a person by uploading their photo
2. Next time they appear, the system recognizes them automatically
3. Attendance is logged with timestamp and exported on demand

---

## Web Application

Built with Streamlit — 3 pages:

- **Take Attendance** — upload or capture a photo for instant recognition
- **Register Face** — add new people to the system
- **View Records** — filter, search, and export attendance data

---

## Tech Stack

- Python 3.12
- DeepFace — face recognition model
- OpenCV — image processing
- Pandas — data handling
- Streamlit — web interface
- Pillow — image manipulation

---

## Getting Started

```bash
git clone https://github.com/Youssefnada-Engineer/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure
face-recognition-attendance/

├── app.py            # Main application

├── known_faces/      # Stored face images

├── attendance.csv    # Attendance log

└── requirements.txt  # Dependencies
---

## Author

Youssef — AI/ML Intern