# Object Detection and Tracking System for Visually Impaired (ODTSVI)

- [Object Detection and Tracking System for Visually Impaired (ODTSVI)](#object-detection-and-tracking-system-for-visually-impaired-odtsvi)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Libraries and Dependencies](#libraries-and-dependencies)
  - [How It Works](#how-it-works)
    - [Object Detection](#object-detection)
    - [Object Tracking](#object-tracking)
    - [LiDAR Integration](#lidar-integration)
    - [Sound Alerts](#sound-alerts)
  - [Credits](#credits)
  - [License](#license)

## Introduction

ODTSVI (Object Detection and Tracking System for Visually Impaired) is a project designed to assist visually impaired individuals in navigating their environment safely. The system uses a Raspberry Pi mini-PC equipped with a camera module and a LiDAR LD19 sensor to create a "sense of sight" for the user. It provides three-dimensional sound alerts to indicate the location and type of obstacles in the user's surroundings.

## Features

- Real-time object detection and tracking using TensorFlow Lite.
- Integration of LiDAR sensor data to estimate object distances.
- Voice-based sound alerts to inform the user about object labels and distances.
- Webcam video streaming and processing for quick feedback.

## Requirements

- Raspberry Pi mini-PC (tested on Raspberry Pi 3B+ and above).
- Camera Module v2 for video input.
- LiDAR LD19 sensor for distance estimation.
- Python 3.x installed on the Raspberry Pi.
- Required libraries and dependencies (see [Libraries and Dependencies](#libraries-and-dependencies)).

## Installation

1. Clone this repository to your Raspberry Pi:

```bash
git clone https://github.com/zaki-x86/LiDAR-object-detection
cd LiDAR-object-detection
```

1. Install the required libraries and dependencies (see [Libraries and Dependencies](#libraries-and-dependencies)).

## Usage

1. Connect the Camera Module v2 and the LiDAR LD19 sensor to the Raspberry Pi.
2. Ensure that all required libraries are installed.
3. Run the `main.py` script to start the Object Detection and Tracking System.

```bash
python3 main.py
```

4. The system will start processing the video stream, detecting objects, and providing sound alerts to the user.

## Project Structure

- `main.py`: The main script that handles video streaming, object detection, tracking, and LiDAR integration.
- `sound.py`: Contains the `SoundFunc` function responsible for playing sound alerts to the user.
- `lidar.py`: Defines the `lidarfunc` function and classes for handling LiDAR data processing.
- `README.md`: This README file.

## Libraries and Dependencies

The following libraries and dependencies are required to run the ODTSVI project:

- [NumPy](https://numpy.org/): For numerical computations and array processing.
- [OpenCV](https://opencv.org/): For computer vision tasks, including video streaming and image processing.
- [TensorFlow Lite](https://www.tensorflow.org/lite): For object detection using pre-trained models.
- [gtts](https://gtts.readthedocs.io/en/latest/): For text-to-speech synthesis.
- [openal](https://pypi.org/project/pyopenal/): For sound playback.

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## How It Works

### Object Detection

The object detection system is based on a pre-trained TensorFlow Lite model. The webcam captures video frames, and the model performs inference on these frames to detect objects. The model provides bounding box coordinates, class labels, and confidence scores for each detected object. Objects with confidence scores above a specified threshold are considered as detected objects.

### Object Tracking

Detected objects are tracked across consecutive frames using OpenCV's `MultiTracker`. When a new object is detected, it is added to the `MultiTracker`, which uses various tracking algorithms to predict the positions of tracked objects in the next frame. This allows the system to continuously monitor and update the positions of objects as they move.

### LiDAR Integration

The LiDAR LD19 sensor provides distance data for objects in the environment. By correlating the detected objects' positions in the frame with the corresponding LiDAR distance data, the system estimates the distance of each object from the camera. This information is crucial for providing accurate sound alerts to the user.

### Sound Alerts

The `SoundFunc` function is responsible for playing voice-based sound alerts to the user. It uses the Google Text-to-Speech (gtts) library to synthesize voice messages. The system provides the user with information about the detected object's label and its estimated distance. The alerts are designed to help the visually impaired user navigate the environment safely.

## Credits

The ODTSVI project is inspired by various computer vision, object detection, and LiDAR integration examples and tutorials available in the open-source community. Special thanks to the contributors and developers of the libraries and dependencies used in this project.

- The object Detection is based off "the Webcam Object Detection Using Tensorflow-trained Classifier" [example](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_webcam.py)

- VideoStream class to handle streaming of video from webcam in separate processing thread Source - [Adrian Rosebrock, PyImageSearch](https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/)

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify it according to your needs.