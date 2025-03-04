# Lab-Works
My works in ITU Soft Sensors Lab as an undergraduate intern
# Angle Calculation with ArUco & Sign Language Recognition

This repository contains code for two main projects:
1. **Angle Calculation using ArUco Markers** - Detects and calculates angles using OpenCV and ArUco markers.
2. **Sign Language Recognition with Smart Glove** - Uses sensor data to classify sign language gestures with machine learning.

## Features
- **ArUco-Based Angle Calculation:**
  - Detects ArUco markers in an image or video.
  - Computes the angle between markers for various applications.
- **Sign Language Recognition:**
  - Collects sensor data from a smart glove.
  - Trains and evaluates a machine learning model for gesture classification.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have OpenCV, NumPy, and other required libraries.

## Usage
### ArUco Angle Calculation
Run the script to detect markers and compute angles:
```bash
python aruco_angle.py
```
Ensure your camera is properly connected or provide an image with ArUco markers.

### Sign Language Recognition
1. **Data Collection:** Use the smart glove to collect sensor data and store it.
2. **Model Training:** Train a machine learning model using the dataset.
3. **Prediction:** Run the model on real-time sensor data:
   ```bash
   python sign_language.py
   ```

## Future Improvements
- Improve gesture classification accuracy.
- Enhance real-time processing speed.
- Develop a user-friendly interface.

## Contributing
Feel free to submit pull requests or open issues for enhancements and bug fixes.

## License
This project is licensed under the MIT License.

