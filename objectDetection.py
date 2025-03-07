import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import deque

class VehicleTracker:
    def __init__(self, video_path, model_path="yolov8n.pt", fov=120, video_width=1920, camera_height=1.5):
        self.FOV = fov
        self.VIDEO_WIDTH = video_width
        self.CAMERA_HEIGHT = camera_height
        self.pixel_angle = self.FOV / self.VIDEO_WIDTH

        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.scale_factor = 0.02  # Calibrate this value
        self.track_history = {}  # Track history for each vehicle
        self.stationary_vehicles = {}
        self.risk_duration = {}
        self.vehicle_ids = {}
        self.next_id = 0
        self.active_risks = {}
        self.dangerous_situations = []
        self.continuous_risk_counts = 0
        self.processed_frames = 0
        self.start_time = time.time()

        # Parameters for tracking and collision detection
        self.MIN_SPEED_THRESHOLD = 0.2  # Lower speed threshold
        self.MIN_FRAMES_STATIONARY = 10
        self.RISK_THRESHOLD = 1.0  # Lower risk duration threshold
        self.TRACK_LENGTH = 20  # Increase track history length
        self.MAX_TRACK_LOSS_FRAMES = 5  # Max frames to keep a lost track

    def pixel_to_real_distance(self, pixel_offset):
        angle = pixel_offset * self.pixel_angle
        angle_rad = math.radians(angle)
        real_distance = math.tan(angle_rad) * self.CAMERA_HEIGHT
        return real_distance

    def get_vehicle_id(self, center_x, center_y, area):
        min_distance = float('inf')
        best_id = None
        for pos, vid in list(self.vehicle_ids.items()):
            dist = math.sqrt((center_x - pos[0])**2 + (center_y - pos[1])**2)
            if dist < min_distance and dist < 50:
                min_distance = dist
                best_id = vid

        if best_id is not None:
            for pos in list(self.vehicle_ids.keys()):
                if self.vehicle_ids[pos] == best_id:
                    del self.vehicle_ids[pos]
                    break
            self.vehicle_ids[(center_x, center_y)] = best_id
            return best_id
        else:
            new_id = f"vehicle_{self.next_id}"
            self.next_id += 1
            self.vehicle_ids[(center_x, center_y)] = new_id
            return new_id

    def calculate_collision_time(self, vehicle_center, vehicle_velocity, my_position, my_velocity):
        dx = vehicle_center[0] - my_position[0]
        dy = vehicle_center[1] - my_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        rel_vx = vehicle_velocity[0] - my_velocity[0]
        rel_vy = vehicle_velocity[1] - my_velocity[1]
        rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)

        if rel_speed == 0:
            return float('inf')

        ttc = distance / rel_speed
        return ttc / (self.fps / 2)  # Convert to seconds

    def process_frame(self, frame):
        results = self.model(frame, classes=[2, 5, 7], verbose=False)
        current_vehicles = {}
        collision_risks = []
        current_risk_vehicles = set()

        for i, det in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            area = width * height

            vehicle_id = self.get_vehicle_id(center_x, center_y, area)
            current_vehicles[vehicle_id] = {
                "center": (center_x, center_y),
                "box": (x1, y1, x2, y2),
                "area": area
            }

            # Update track history
            if vehicle_id in self.track_history:
                if len(self.track_history[vehicle_id]) >= self.TRACK_LENGTH:
                    self.track_history[vehicle_id].popleft()
                self.track_history[vehicle_id].append((center_x, center_y))
            else:
                self.track_history[vehicle_id] = deque(maxlen=self.TRACK_LENGTH)
                self.track_history[vehicle_id].append((center_x, center_y))

            # Calculate velocity and speed
            if len(self.track_history[vehicle_id]) >= 2:
                prev_x, prev_y = self.track_history[vehicle_id][-2]
                dx = center_x - prev_x
                dy = center_y - prev_y
                speed_pixels = math.sqrt(dx**2 + dy**2)
                time_delta = 1 / self.fps
                real_speed = (speed_pixels * self.scale_factor) / time_delta

                current_vehicles[vehicle_id].update({
                    "velocity": (dx, dy),
                    "speed": speed_pixels,
                    "real_speed": real_speed
                })

        # My vehicle's position and velocity
        my_vehicle_position = (frame.shape[1] // 2, frame.shape[0] - 20)
        my_velocity = (0, -10)

        # Check for collision risks
        for vehicle_id, vehicle_data in current_vehicles.items():
            if "velocity" not in vehicle_data:
                continue

            vehicle_center = vehicle_data["center"]
            vehicle_velocity = vehicle_data["velocity"]
            vehicle_real_speed = vehicle_data.get("real_speed", 0)

            # Skip stationary vehicles
            if vehicle_real_speed < self.MIN_SPEED_THRESHOLD:
                continue

            # Calculate collision time
            ttc = self.calculate_collision_time(vehicle_center, vehicle_velocity, my_vehicle_position, my_velocity)

            # Determine risk level based on TTC
            risk_level = 0
            if ttc < 1.0:
                risk_level += 3
            elif ttc < 2.0:
                risk_level += 2
            elif ttc < 3.0:
                risk_level += 1

            if risk_level >= 3:
                if vehicle_id in self.risk_duration:
                    self.risk_duration[vehicle_id] += 1 / self.fps
                else:
                    self.risk_duration[vehicle_id] = 1 / self.fps

                if vehicle_id not in self.active_risks and self.risk_duration[vehicle_id] >= self.RISK_THRESHOLD:
                    self.active_risks[vehicle_id] = self.processed_frames / self.fps

                current_risk_vehicles.add(vehicle_id)
                collision_risks.append((vehicle_id, risk_level, vehicle_real_speed, ttc))

                # Debugging output
                print(f"Vehicle ID: {vehicle_id}, TTC: {ttc:.2f}s, Risk Level: {risk_level}, Distance: {math.sqrt((vehicle_center[0] - my_vehicle_position[0])**2 + (vehicle_center[1] - my_vehicle_position[1])**2) * self.scale_factor:.2f}m")

        # Handle lost tracks
        for vehicle_id in list(self.track_history.keys()):
            if vehicle_id not in current_vehicles:
                if len(self.track_history[vehicle_id]) > 0:
                    # Predict position using last known velocity
                    last_center = self.track_history[vehicle_id][-1]
                    if "velocity" in current_vehicles.get(vehicle_id, {}):
                        velocity = current_vehicles[vehicle_id]["velocity"]
                        predicted_center = (last_center[0] + velocity[0], last_center[1] + velocity[1])
                        self.track_history[vehicle_id].append(predicted_center)

                    # Remove track if lost for too long
                    if len(self.track_history[vehicle_id]) > self.MAX_TRACK_LOSS_FRAMES:
                        del self.track_history[vehicle_id]

        # Visualize collision risks
        for vehicle_id, risk_level, rel_speed, ttc in collision_risks:
            if vehicle_id not in current_vehicles:
                continue

            vehicle_center = current_vehicles[vehicle_id]["center"]
            current_risk_duration = self.risk_duration.get(vehicle_id, 0)

            if current_risk_duration >= self.RISK_THRESHOLD:
                color = (0, 0, 255)
                thickness = 3
            elif risk_level >= 5:
                color = (0, 50, 255)
                thickness = 2
            elif risk_level >= 4:
                color = (0, 127, 255)
                thickness = 2
            else:
                color = (0, 165, 255)
                thickness = 2

            cv2.line(frame, my_vehicle_position, vehicle_center, color, thickness)
            mid_x = (my_vehicle_position[0] + vehicle_center[0]) // 2
            mid_y = (my_vehicle_position[1] + vehicle_center[1]) // 2

            if current_risk_duration >= self.RISK_THRESHOLD:
                warning = "CONTINUOUS RISK!"
            elif ttc < 1.5:
                warning = "IMMINENT COLLISION!"
            else:
                warning = "COLLISION RISK"

            cv2.putText(frame, warning, (mid_x - 60, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Risk: {current_risk_duration:.1f}s", (vehicle_center[0], vehicle_center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Speed: {rel_speed:.1f} km/h", (vehicle_center[0], vehicle_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if ttc < float('inf'):
                cv2.putText(frame, f"TTC: {ttc:.1f}s", (vehicle_center[0], vehicle_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display general information
        if collision_risks:
            cv2.putText(frame, "COLLISION RISK DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Continuous Risks: {self.continuous_risk_counts}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"SCALE: {self.scale_factor:.4f} m/px", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.processed_frames % 10 == 0:
            elapsed_time = time.time() - self.start_time
            current_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"Time: {self.processed_frames / self.fps:.1f}s", (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Collision Detection", frame)

    def run(self):
        paused = False
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.processed_frames += 1

            if self.processed_frames % 2 != 0:
                continue

            self.process_frame(frame)

            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key == ord('w'):
                paused = not paused
                if paused:
                    print("Video stopped")
                else:
                    print("Video continue.")
            elif key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\nÖzet Rapor:")
        print(f"Toplam tespit edilen sürekli risk durumu: {self.continuous_risk_counts}")
        print(f"İşlenen toplam kare: {self.processed_frames}")
        print(f"Toplam süre: {time.time() - self.start_time:.2f} saniye")

        if self.dangerous_situations:
            print("\nTehlikeli Durumlar:")
            for i, situation in enumerate(self.dangerous_situations, 1):
                print(f"{i}. Araç ID: {situation['vehicle_id']}, " +
                      f"Başlangıç: {situation['start_time']:.2f}s, " +
                      f"Bitiş: {situation['end_time']:.2f}s, " +
                      f"Süre: {situation['duration']:.2f}s")

if __name__ == "__main__":
    tracker = VehicleTracker("videos/sample02.webm")
    tracker.run()