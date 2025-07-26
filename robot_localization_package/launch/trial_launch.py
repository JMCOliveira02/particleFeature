import os
import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

def generate_launch_description():
    trial_type = "Nave_A_MM"

    # Waypoint Follower - using the newer trajectory
    waypoint_file = f"/home/biltes/ros_ws/src/robot_waypoint_follower/robot_waypoint_follower/Nave_A_MM_new.yaml"
    waypoint_follower = Node(
        package='robot_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        parameters=[{
            'waypoints_file': waypoint_file,
            'linear_speed': 0.26,
            'angular_speed': 0.4,
            'position_threshold': 0.2,
            'orientation_threshold': 0.3,
            'strict_final_orientation': True,
        }], 
        output='screen'
    )

        # Record and playback trajectory (not used)
    ##Path Tracker
    path_tracker = Node(
        package='robot_worlds',
        executable='path_tracker',
        name='path_tracker',
        parameters=[{'trial_type' : trial_type}], 
        output='screen'
    )

    return LaunchDescription([
        path_tracker,
        waypoint_follower
    ])