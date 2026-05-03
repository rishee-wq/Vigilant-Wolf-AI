This README is designed to make your GitHub repository look professional, high-impact, and technically robust. It highlights the unique engineering choices you made for **Vigilant Wolf AI**, such as the 30 FPS processing and the sub-50ms latency.

---

# 🐺 Vigilant Wolf AI
**Real-Time Driver Fatigue Detection System**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/Powered%20By-MediaPipe-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
**Vigilant Wolf AI** is a high-performance computer vision solution designed to prevent road accidents caused by driver drowsiness[cite: 1]. Unlike reactive safety measures that trigger only after a vehicle drifts, this system monitors the driver's physiological state in real-time using a 468-point 3D facial mesh[cite: 1]. 

By calculating the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**, the system detects microsleeps and yawning with sub-millimeter precision[cite: 1].



## ⚡ Key Technical Features
*   **3D Facial Mesh Engine**: Leverages Google's MediaPipe for precise tracking of eyelid and lip movement[cite: 1].
*   **Ultra-Low Latency**: A detection-to-alert pipeline optimized to under **50ms**[cite: 1].
*   **High-Speed Processing**: Runs consistently at **30+ FPS**, ensuring no critical blink or microsleep is missed[cite: 1].
*   **Multi-Modal Intervention**: Features auditory alarms, visual danger overlays, and a 10-second emergency auto-call countdown[cite: 1].
*   **Ensemble ML Validation**: Backed by a theoretical validation accuracy of **99%**[cite: 1].

## 🛠️ Tech Stack
*   **Core**: Python
*   **Computer Vision**: MediaPipe, OpenCV
*   **GUI**: PyQt5
*   **Machine Learning**: Scikit-Learn, Keras (for validation)
*   **Environment**: Optimized for cross-platform deployment (Windows/Linux)[cite: 1].

## 🚀 Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/RisheeSharma/Vigilant-Wolf-AI.git
   cd Vigilant-Wolf-AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

## 📐 How It Works
The system monitors two primary geometric ratios:
1.  **EAR (Eye Aspect Ratio)**: Triggers an alarm when the ratio drops below **0.21** for a sustained period[cite: 1].
2.  **MAR (Mouth Aspect Ratio)**: Detects yawning patterns to provide early-stage fatigue warnings[cite: 1].



## 📊 Performance Analysis
In experimental settings using a **Tesla GPU T4** environment, our Ensemble Voting strategy achieved:
*   **Accuracy**: 99.0%[cite: 1]
*   **Sensitivity**: 96.8%[cite: 1]
*   **Latency**: <50ms[cite: 1]

## 🔮 Future Roadmap
*   **Gaze Tracking**: Identifying when the driver's attention shifts away from the road[cite: 1].
*   **Night Vision Optimization**: Utilizing Histogram Equalization for pitch-black cabin environments[cite: 1].
*   **Edge Hardware Integration**: Porting the engine to Raspberry Pi/NVIDIA Jetson for in-vehicle deployment.


---
**Author:** Rishee Sharma  
**Project:** B.Tech CSE - 2nd Year (UPES)
```
