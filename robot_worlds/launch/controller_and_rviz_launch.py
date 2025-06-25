import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import OpaqueFunction


def generate_launch_description():
    world_dir = get_package_share_directory('robot_worlds')
    robot_description_path = os.path.join(world_dir, 'urdf', 'robot.urdf')
    world_setup = "iilab"

    robot_controller = WebotsController(
        robot_name='robot',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    webots = WebotsLauncher(
        world=os.path.join(world_dir, 'worlds', world_setup, world_setup + '.wbt')
    )

    teleop = Node(
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        name="teleop_twist_keyboard",
        output="screen",
        prefix="gnome-terminal --",
    )

    tf_static_lidar = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_broadcaster",
        arguments=["0", "0", "0", "0", "0", "0", "base_footprint", "lidar2D"]
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(world_dir, "rviz", "tf_and_map.rviz")],
        output="screen"
    )


    return LaunchDescription([
        rviz,
        tf_static_lidar,
        webots,
        robot_controller,
        teleop,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])