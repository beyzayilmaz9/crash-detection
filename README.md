# Vehicle Collision Detection System

## Overview
The **Vehicle Collision Detection System** is an advanced real-time object detection and tracking tool designed to analyze vehicle movements from a video feed and predict potential collisions. This system leverages the **YOLOv8** deep learning model to detect vehicles and compute their trajectories to assess risks.

## Features
- **Real-time vehicle detection** using YOLOv8.
- **Risk assessment and collision prediction** based on vehicle speed and trajectory.
- **Imminent collision warnings** for enhanced safety.
- **Continuous risk detection** for ongoing hazardous situations.
- **Customizable parameters** such as camera field of view (FOV) and height.
- **Comprehensive logs and visual alerts** for detected risks.

---

## Requirements
Ensure you have the following dependencies installed:

### Python Packages
```
pip install ultralytics opencv-python numpy
```

### System Requirements
- Python 3.8 or higher
- OpenCV (for image processing and video handling)
- A compatible GPU (recommended for optimal performance)

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure YOLOv8 model weights are available. The default model used is `yolov8n.pt`. You can download different versions from:
   ```
   https://github.com/ultralytics/ultralytics
   ```

---

## Usage
Run the script with the following command:
```bash
python objectDetection.py --input path_to_video.mp4 --model yolov8n.pt --fov 120 --height 1.5
```

### Command Line Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input video file (Required) | N/A |
| `--model` | Path to YOLOv8 model file | `yolov8n.pt` |
| `--fov` | Camera field of view in degrees | `120` |
| `--height` | Camera height in meters | `1.5` |

---

## How It Works
1. **Video Processing:** The system reads video frames and processes them using the YOLO model.
2. **Vehicle Tracking:** Identifies and assigns unique IDs to vehicles.
3. **Trajectory Analysis:** Computes speed, direction, and distance to predict collision risks.
4. **Risk Assessment:** Evaluates potential collision threats based on velocity and time-to-collision calculations.
5. **Continuous and Imminent Risk Detection:** 
   - **Continuous Risk:** When a vehicle consistently poses a threat over a prolonged duration, the system marks it as a continuous risk.
   - **Imminent Risk:** If a vehicle is on a direct collision course with very little time-to-collision (TTC), the system raises an imminent risk alert.
6. **Visual Alerts:** Displays warnings on the video feed when high-risk scenarios are detected.

![image](https://github.com/user-attachments/assets/90491e84-72e4-43d1-84e4-ebf2a858d04f)

![image](https://github.com/user-attachments/assets/03f0e063-3e16-41ce-a859-4e8835f89e7e)

---

## Expected Output
- **Frame-by-frame risk visualization**
- **Real-time collision warnings**
- **Continuous and imminent risk notifications**
- **Terminal logs with detected risks and vehicle statistics**

You can add images/screenshots to visualize sample outputs.

---

## Known Issues & Limitations
- **Environmental Factors:** Poor lighting or obstructions may affect detection accuracy.
- **High Processing Demand:** Real-time analysis may require a strong GPU for optimal performance.
- **Calibration Required:** The system's accuracy depends on correct field of view (FOV) and camera height calibration.
- **Speed Estimation Adjustments:** The detected speed may vary depending on camera placement and scale factors. If necessary, fine-tune the scale factor in the script to improve accuracy.

---


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


### Acknowledgments
This project uses **Ultralytics YOLOv8** for object detection.

