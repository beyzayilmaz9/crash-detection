import cv2
import numpy as np
import math
import time
from collections import deque
import argparse
import sys
import os

def check_dependencies():
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        print("Error: The ultralytics package is not installed.")
        print("Please install it using: pip install ultralytics")
        return False

class VehicleTracker:
    def __init__(self, video_path, model_path="yolov8n.pt", fov=120, video_width=1920, camera_height=1.5):
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' does not exist.")
            sys.exit(1)
            
        # Import YOLO here to handle potential import errors gracefully
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Error: Failed to import YOLO. Please ensure ultralytics is installed.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            sys.exit(1)
            
        # Initialize video capture with error handling
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'.")
            sys.exit(1)
            
        self.FOV = fov
        self.VIDEO_WIDTH = video_width
        self.CAMERA_HEIGHT = camera_height
        
        # Get video properties with safeguards
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            print("Warning: Invalid FPS detected, using default of 30.")
            self.fps = 30
            
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.actual_width <= 0:
            print("Warning: Invalid video width detected, using default.")
            self.actual_width = video_width
            
        # Calculate pixel angle
        self.pixel_angle = self.FOV / self.actual_width
        
        # Calibration settings
        self.scale_factor = 0.002  # Calibrated for realistic speeds
        
        # Tracking data structures
        self.track_history = {}
        self.stationary_vehicles = {}
        self.risk_duration = {}
        self.vehicle_ids = {}
        self.next_id = 0
        self.active_risks = {}
        self.dangerous_situations = []
        self.continuous_risk_counts = 0
        self.processed_frames = 0
        self.start_time = time.time()
        self.lost_track_counter = {}
        
        # Detection parameters
        self.MIN_SPEED_THRESHOLD = 0.2
        self.MIN_FRAMES_STATIONARY = 10
        
        # MODIFIED: Lowered threshold for active risk detection to make risks appear sooner
        self.RISK_THRESHOLD = 0.2  # Reduced from 0.6 to register risks faster
        
        self.TRACK_LENGTH = 20
        self.MAX_TRACK_LOSS_FRAMES = 5
        
        # ADDED: New parameters for imminent collision handling
        self.IMMINENT_COLLISION_TTC = 2.0  # Consider collisions imminent below this TTC
        self.imminent_collisions = {}  # Track imminent collisions separately
        
        print(f"Initialized VehicleTracker with video: {video_path}")
        print(f"Model: {model_path}, FOV: {fov}°, Camera Height: {camera_height}m")
        print(f"Video FPS: {self.fps:.2f}, Width: {self.actual_width}px")

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
        # Calculate relative position
        dx = vehicle_center[0] - my_position[0]
        dy = vehicle_center[1] - my_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate relative velocity
        rel_vx = vehicle_velocity[0] - my_velocity[0]
        rel_vy = vehicle_velocity[1] - my_velocity[1]
        rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)
        
        # Avoid division by very small numbers
        if rel_speed < 0.001:
            return float('inf')
            
        # IMPROVED: Better detection for vehicles approaching from sides
        # Use projected closest approach distance instead of just current direction
        # Project position forward based on velocity vectors
        closest_approach_dist = self.calculate_closest_approach(dx, dy, rel_vx, rel_vy)
        
        # If vehicles will pass close to each other, consider it a potential collision
        if closest_approach_dist < 100:  # Threshold in pixels for "near miss"
            # Time to collision = distance / relative speed
            ttc = distance / rel_speed
            
            # Convert to seconds based on video fps
            return ttc / (self.fps / 2)
        else:
            return float('inf')

    def calculate_closest_approach(self, dx, dy, rel_vx, rel_vy):
        """Calculate the closest approach distance between two vehicles based on their current positions and velocities"""
        # If velocities are nearly zero, return current distance
        if abs(rel_vx) < 0.001 and abs(rel_vy) < 0.001:
            return math.sqrt(dx*dx + dy*dy)
        
        # Time of closest approach
        t = -(dx*rel_vx + dy*rel_vy) / (rel_vx*rel_vx + rel_vy*rel_vy)
        
        # If closest approach is in the past, use current distance
        if t < 0:
            return math.sqrt(dx*dx + dy*dy)
        
        # Calculate position at closest approach
        closest_dx = dx + rel_vx * t
        closest_dy = dy + rel_vy * t
        
        # Return the distance at closest approach
        return math.sqrt(closest_dx*closest_dx + closest_dy*closest_dy)
    def process_frame(self, frame):
        try:
            # Run object detection
            results = self.model(frame, classes=[2, 5, 7], verbose=False)
            
            current_vehicles = {}
            collision_risks = []
            current_risk_vehicles = set()
            current_imminent_vehicles = set()  # ADDED: Track imminent collision vehicles
            
            # Store frame dimensions for reference
            frame_height, frame_width = frame.shape[:2]
            
            # Process detection results
            if len(results) > 0 and hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'data'):
                for i, det in enumerate(results[0].boxes.data):
                    if len(det) < 6:  # Ensure we have all needed values
                        continue
                        
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
    
                    # Reset lost track counter for this vehicle
                    self.lost_track_counter[vehicle_id] = 0
    
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
                        
                        # Convert pixel speed to real-world speed (km/h)
                        real_speed = (speed_pixels * self.scale_factor * self.fps * 3.6)
                        
                        # Apply distance correction based on y position
                        y_position_factor = 1.0 + (center_y / frame_height) * 0.5
                        real_speed = real_speed * y_position_factor
                        
                        # Cap speed at realistic values
                        real_speed = min(real_speed, 150.0)
    
                        current_vehicles[vehicle_id].update({
                            "velocity": (dx, dy),
                            "speed": speed_pixels,
                            "real_speed": real_speed
                        })

            # Increment lost track counter for vehicles not in current frame
            for vehicle_id in self.track_history:
                if vehicle_id not in current_vehicles:
                    if vehicle_id in self.lost_track_counter:
                        self.lost_track_counter[vehicle_id] += 1
                    else:
                        self.lost_track_counter[vehicle_id] = 1
    
            # My vehicle's position and velocity
            my_vehicle_position = (frame.shape[1] // 2, frame.shape[0] - 20)
            my_velocity = (0, -5)  # Reduced for more realistic calculations
    
            # Check for collision risks
            for vehicle_id, vehicle_data in current_vehicles.items():
                if "velocity" not in vehicle_data:
                    continue
    
                vehicle_center = vehicle_data["center"]
                vehicle_velocity = vehicle_data["velocity"]
                vehicle_real_speed = vehicle_data.get("real_speed", 0)
    
                # Skip truly stationary vehicles
                if vehicle_real_speed < self.MIN_SPEED_THRESHOLD:
                    continue
    
                # Calculate collision time
                ttc = self.calculate_collision_time(vehicle_center, vehicle_velocity, my_vehicle_position, my_velocity)
    
                # Determine risk level based on TTC
                risk_level = 0
                if ttc < 1.0:
                    risk_level = 3  # High risk
                elif ttc < 2.0:
                    risk_level = 2  # Medium risk
                elif ttc < 3.0:
                    risk_level = 1  # Low risk
    
                # ADDED: Immediately register imminent collisions
                if ttc < self.IMMINENT_COLLISION_TTC:
                    if vehicle_id not in self.imminent_collisions:
                        print(f"Imminent collision detected: {vehicle_id} at {self.processed_frames / self.fps:.2f}s, TTC: {ttc:.2f}s")
                        self.imminent_collisions[vehicle_id] = {
                            'start_time': self.processed_frames / self.fps,
                            'ttc': ttc
                        }
                    current_imminent_vehicles.add(vehicle_id)
                    
                    # MODIFIED: Ensure imminent collisions are immediately registered as active risks
                   # Before adding to dangerous situations, check if this is truly a new risk
                    if vehicle_id not in self.active_risks:
                        self.active_risks[vehicle_id] = self.processed_frames / self.fps
                        # Only increment counter for new unique vehicles, not for the same vehicle reappearing
                        if vehicle_id not in [situation['vehicle_id'] for situation in self.dangerous_situations]:
                            self.continuous_risk_counts += 1
                            
                        # Debug print for continuous risk
                        print(f"New continuous risk detected: {vehicle_id} at {self.processed_frames / self.fps:.2f}s")
                        
                        # Add to dangerous situations
                        self.dangerous_situations.append({
                            'vehicle_id': vehicle_id,
                            'start_time': self.processed_frames / self.fps,
                            'end_time': None,
                            'duration': 0,
                            'type': 'continuous'
                        })
    
                # Track vehicles with significant risk
                if risk_level >= 1:
                    if vehicle_id in self.risk_duration:
                        self.risk_duration[vehicle_id] += 1 / self.fps
                    else:
                        self.risk_duration[vehicle_id] = 1 / self.fps
    
                    # Check for continuous risk
                    if self.risk_duration[vehicle_id] >= self.RISK_THRESHOLD:
                        if vehicle_id not in self.active_risks:
                            self.active_risks[vehicle_id] = self.processed_frames / self.fps
                            self.continuous_risk_counts += 1
                            
                            # Debug print for continuous risk
                            print(f"New continuous risk detected: {vehicle_id} at {self.processed_frames / self.fps:.2f}s")
                            
                            # Add to dangerous situations
                            self.dangerous_situations.append({
                                'vehicle_id': vehicle_id,
                                'start_time': self.processed_frames / self.fps,
                                'end_time': None,
                                'duration': 0,
                                'type': 'continuous'
                            })
    
                    current_risk_vehicles.add(vehicle_id)
                    collision_risks.append((vehicle_id, risk_level, vehicle_real_speed, ttc))
                else:
                    # Gradually decrease risk duration if risk level is low
                    if vehicle_id in self.risk_duration:
                        self.risk_duration[vehicle_id] -= 0.3 / self.fps
                        if self.risk_duration[vehicle_id] <= 0:
                            self.risk_duration[vehicle_id] = 0
            
            # Update end times for resolved risks
            for vehicle_id in list(self.active_risks.keys()):
                if vehicle_id not in current_risk_vehicles and vehicle_id not in current_imminent_vehicles:
                    # Only end the risk if the vehicle is truly gone (not just temporarily lost)
                    if vehicle_id not in current_vehicles and self.lost_track_counter.get(vehicle_id, 0) > 2:
                        # Find the dangerous situation and update it
                        for situation in self.dangerous_situations:
                            if situation['vehicle_id'] == vehicle_id and situation['end_time'] is None:
                                situation['end_time'] = self.processed_frames / self.fps
                                situation['duration'] = situation['end_time'] - situation['start_time']
                                
                                # Debug print for risk ending
                                print(f"Risk ended: {vehicle_id}, duration: {situation['duration']:.2f}s")
                                break
                        
                        # Remove from active risks
                        del self.active_risks[vehicle_id]
                        
                        # Also remove from imminent collisions if present
                        if vehicle_id in self.imminent_collisions:
                            del self.imminent_collisions[vehicle_id]
    
            # Handle lost tracks
            for vehicle_id in list(self.track_history.keys()):
                if vehicle_id not in current_vehicles:
                    # Remove track if lost for too long
                    if self.lost_track_counter.get(vehicle_id, 0) > self.MAX_TRACK_LOSS_FRAMES:
                        del self.track_history[vehicle_id]
                        if vehicle_id in self.lost_track_counter:
                            del self.lost_track_counter[vehicle_id]
                        
                        # Also ensure it's removed from risk tracking if needed
                        if vehicle_id in self.risk_duration and vehicle_id not in self.active_risks:
                            del self.risk_duration[vehicle_id]
    
            # VISUALIZATION
            # Visualize collision risks
            for vehicle_id, risk_level, real_speed, ttc in collision_risks:
                if vehicle_id not in current_vehicles:
                    continue
    
                vehicle_center = current_vehicles[vehicle_id]["center"]
                current_risk_duration = self.risk_duration.get(vehicle_id, 0)
    
                # MODIFIED: Color coding based on multiple factors
                if vehicle_id in self.imminent_collisions:
                    color = (0, 0, 255)  # Red for imminent collision
                    thickness = 3
                elif current_risk_duration >= self.RISK_THRESHOLD:
                    color = (0, 0, 255)  # Red for continuous risk
                    thickness = 3
                elif risk_level >= 2:
                    color = (0, 50, 255)  # Orange-ish for medium risk
                    thickness = 2
                else:
                    color = (0, 165, 255)  # Yellow-ish for low risk
                    thickness = 2
    
                # Draw line from my position to vehicle
                cv2.line(frame, my_vehicle_position, vehicle_center, color, thickness)
                mid_x = (my_vehicle_position[0] + vehicle_center[0]) // 2
                mid_y = (my_vehicle_position[1] + vehicle_center[1]) // 2
    
                # Display appropriate warning text
                if vehicle_id in self.imminent_collisions:
                    warning = "IMMINENT COLLISION!"
                elif current_risk_duration >= self.RISK_THRESHOLD:
                    warning = "CONTINUOUS RISK!"
                elif ttc < 1.5:
                    warning = "IMMINENT COLLISION!"
                else:
                    warning = "COLLISION RISK"
    
                cv2.putText(frame, warning, (mid_x - 60, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Risk: {current_risk_duration:.1f}s", (vehicle_center[0], vehicle_center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Speed: {real_speed:.1f} km/h", (vehicle_center[0], vehicle_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if ttc < float('inf'):
                    cv2.putText(frame, f"TTC: {ttc:.1f}s", (vehicle_center[0], vehicle_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
            # Display general information
            if collision_risks:
                cv2.putText(frame, "COLLISION RISK DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            cv2.putText(frame, f"Continuous Risks: {self.continuous_risk_counts}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Active Risks: {len(self.active_risks)}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ADDED: Show imminent collisions count
            cv2.putText(frame, f"Imminent Collisions: {len(self.imminent_collisions)}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"SCALE: {self.scale_factor:.4f} m/px", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
            # Only update FPS display occasionally
            if self.processed_frames % 10 == 0:
                elapsed_time = time.time() - self.start_time
                current_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
            cv2.putText(frame, f"Time: {self.processed_frames / self.fps:.1f}s", (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Debug info
            cv2.putText(frame, f"Vehicles: {len(current_vehicles)}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Collision Detection", frame)
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            import traceback
            traceback.print_exc()

    def run(self):
        try:
            paused = False
            print("Starting video processing. Press 'w' to pause/resume, ESC to exit.")
            
            while self.cap.isOpened():
                # Read frame with error handling
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached.")
                    break

                self.processed_frames += 1

                # Process every other frame for performance
                if self.processed_frames % 2 != 0:
                    continue
                    
                # Process current frame
                self.process_frame(frame)

                # Handle key presses
                key = cv2.waitKey(1 if not paused else 0) & 0xFF

                if key == ord('w'):
                    paused = not paused
                    if paused:
                        print("Video paused")
                    else:
                        print("Video resumed")
                elif key == 27:  # ESC key
                    print("ESC pressed, exiting.")
                    break

            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print summary report
            self.print_summary()
            
        except Exception as e:
            print(f"Error in run method: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure resources are released
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            
    def print_summary(self):
        print(f"\nSummary Report:")
        print(f"Total continuous risk situations detected: {self.continuous_risk_counts}")
        print(f"Total frames processed: {self.processed_frames}")
        print(f"Total processing time: {time.time() - self.start_time:.2f} seconds")

        if self.dangerous_situations:
            print("\nDangerous Situations:")
            for i, situation in enumerate(self.dangerous_situations, 1):
                end_time = situation['end_time'] if situation['end_time'] is not None else self.processed_frames / self.fps
                duration = end_time - situation['start_time']
                risk_type = situation.get('type', 'standard')
                print(f"{i}. Vehicle ID: {situation['vehicle_id']}, " +
                      f"Type: {risk_type}, " +
                      f"Start: {situation['start_time']:.2f}s, " +
                      f"End: {end_time:.2f}s, " +
                      f"Duration: {duration:.2f}s")
        else:
            print("\nNo dangerous situations detected.")


def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle Collision Detection System')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', help='Path to YOLOv8 model file')
    parser.add_argument('--fov', type=float, default=120, help='Camera field of view in degrees')
    parser.add_argument('--height', type=float, default=1.5, help='Camera height in meters')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        print("Vehicle Collision Detection System")
        print("==================================")
        
        # Check dependencies first
        if not check_dependencies():
            sys.exit(1)
            
        # Parse command line arguments
        args = parse_args()
        
        # Print startup info
        print(f"Input video: {args.input}")
        print(f"Model: {args.model}")
        print(f"FOV: {args.fov}°")
        print(f"Camera height: {args.height}m")
        
        # Initialize and run tracker
        tracker = VehicleTracker(
            video_path=args.input, 
            model_path=args.model,
            fov=args.fov,
            camera_height=args.height
        )
        tracker.run()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
