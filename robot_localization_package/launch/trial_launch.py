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
    trial_type = "Nave_A_AMCL"

    # Waypoint Follower
    waypoint_file = f"/home/joao/ros2_ws/src/robot_waypoint_follower/robot_waypoint_follower/{trial_type}.yaml"
    waypoint_follower = Node(
        package='robot_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        parameters=[{'waypoints_file' : waypoint_file}], 
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