# Lane Detection System

This project implements a **Lane Detection System** using computer vision techniques. The system identifies lane lines on road images or videos, which is a critical component in autonomous driving systems. The pipeline processes images and videos to detect and overlay lane lines using edge detection, region masking, and the Hough Transform.



---

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)


---

## Features
- Detect lane lines in road images.
- Process video streams for real-time lane detection.
- Visualize intermediate steps of the pipeline (e.g., edge detection, region masking).
- Modular code structure for easy modifications.

---

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: Image processing and computer vision.
- **NumPy**: Numerical computations.
- **Matplotlib**: Visualization of results.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lane-detection.git
   cd lane-detection
2. Install dependencies:
   pip install -r requirements.txt
3. Download or add a test image or video in the project directory.
   Image should have the name "lane.jpg". Video should have the name "labe_video.mp4"

##  Usage
1. Run the script for image and video processing
   ```bash
   python main.py
2. The output images and frames will show detected lanes overlaid on the original content.

## How it works
The system utilizes OpenCV with the follwing pipeline:
1. **Grayscale Conversion**: Converts the input image to grayscale for simpler processing.
2. **Gaussian Blur**: Reduces noise for smoother edge detection.
3. **Canny Edge Detection**: Identifies edges in the image.
4. **Region Masking**: Focuses on the region of interest (typically a triangular section).
5. **Hough Transform**: Detects lines in the edge-detected image.
6. **Averaging**: Computes average slopes and intercepts for lane lines.
7. **Overlay**: Draws detected lanes on the original image.

## Future Improvements
1. Add support for curved lane detection.
2. Enhance real-time performance with optimizations.
3. Extend compatibility to various road conditions (e.g., rainy, snowy, night driving).
4. Integrate machine learning models for robust lane segmentation.

