import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    world_dir = get_package_share_directory('robot_worlds')
    robot_description_path = os.path.join(world_dir, 'urdf', 'robot.urdf')

    robot_controller = WebotsController(
        robot_name='robot',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    teleop = Node(
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        name="teleop_twist_keyboard",
        output="screen",
        prefix="gnome-terminal --",
    )

    tf_static = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]
    )

    tf_static_rgbd = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0", "0", "0", "0", "-0.5", "0", "base_footprint", "rgbd"]
    )

    tf_static_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "lidar2D"]
    )


    return LaunchDescription([
        tf_static,
        tf_static_lidar,
        tf_static_rgbd,
        robot_controller,
        teleop
    ])