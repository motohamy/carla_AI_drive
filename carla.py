import carla
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import time
import cv2

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class CarlaEnvironment:
    def __init__(self, town='Town01', port=2000):
        # Connect to CARLA server
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # Initialize variables
        self.vehicle = None
        self.sensors = []
        self.sensor_data = {}
        
        # State and action dimensions
        self.state_dim = 20  # Customize based on your needs
        self.action_dim = 2  # [steering, throttle]
        
    def reset(self):
        """Reset environment and return initial state"""
        # Clean up existing actors
        if self.vehicle:
            for sensor in self.sensors:
                sensor.destroy()
            self.vehicle.destroy()
        
        self.sensors = []
        self.sensor_data = {}
        
        # Spawn vehicle
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Setup sensors
        self._setup_sensors()
        
        # Get initial state
        state = self._get_state()
        return torch.FloatTensor(state)
    
    def _setup_sensors(self):
        """Setup required sensors"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Camera setup
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._process_camera_data(image))
        self.sensors.append(camera)
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        collision_sensor.listen(lambda event: self._process_collision(event))
        self.sensors.append(collision_sensor)
    
    def _get_state(self):
        """Get current state observation"""
        if not self.vehicle:
            return np.zeros(self.state_dim)
        
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        acceleration = self.vehicle.get_acceleration()
        
        # Compile state vector
        state = np.zeros(self.state_dim)
        state[0:3] = [transform.location.x, transform.location.y, transform.rotation.yaw]
        state[3:6] = [velocity.x, velocity.y, velocity.z]
        state[6:9] = [acceleration.x, acceleration.y, acceleration.z]
        
        # Add additional state information (customize as needed)
        # e.g., distance to waypoints, traffic lights, other vehicles
        
        return state
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        if not self.vehicle:
            return torch.zeros(self.state_dim), 0.0, True
        
        # Convert action tensor to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Apply vehicle control
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(-action[1], 0.0, 1.0)) if action[1] < 0 else 0.0
        
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        
        # Simulate one step
        self.world.tick()
        
        # Get new state and reward
        next_state = self._get_state()
        reward = self._compute_reward()
        done = self._is_done()
        
        return torch.FloatTensor(next_state), reward, done
    
    def _compute_reward(self):
        """Compute reward based on current state"""
        reward = 0.0
        
        if not self.vehicle:
            return reward
        
        # Speed reward
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        reward += speed * 0.1  # Reward for maintaining speed
        
        # Collision penalty
        if hasattr(self, 'collision_detected') and self.collision_detected:
            reward -= 100.0
            self.collision_detected = False
        
        # Lane keeping reward
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        if waypoint:
            distance_from_center = waypoint.transform.location.distance(self.vehicle.get_location())
            reward -= distance_from_center * 0.1  # Penalty for deviating from lane center
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        if not self.vehicle:
            return True
        
        # Check for collision
        if hasattr(self, 'collision_detected') and self.collision_detected:
            return True
        
        # Check if vehicle is stuck
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        if speed < 0.1:  # Vehicle is considered stuck if speed is too low
            return True
        
        return False
    
    def _process_camera_data(self, image):
        """Process camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.sensor_data['camera'] = array
    
    def _process_collision(self, event):
        """Process collision event"""
        self.collision_detected = True
    
    def close(self):
        """Cleanup environment"""
        if self.vehicle:
            for sensor in self.sensors:
                sensor.destroy()
            self.vehicle.destroy()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)