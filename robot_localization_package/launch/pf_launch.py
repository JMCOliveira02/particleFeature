import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown


def generate_launch_description():
    # Paths from both packages
    localization_dir = get_package_share_directory(
        'robot_localization_package')
    worlds_dir = get_package_share_directory('robot_worlds')

    # World setup
    world_setup = "iilab"
    # Paths to files
    robot_urdf = os.path.join(worlds_dir, 'urdf', 'robot.urdf')
    world_file = os.path.join(worlds_dir, 'worlds',
                              world_setup, world_setup + '.wbt')
    map_yaml = os.path.join(
        worlds_dir, 'maps', world_setup, world_setup + '.yaml')
    map_pgm = os.path.join(
        worlds_dir, 'maps', world_setup, world_setup + '.pgm')
    map_features = os.path.join(
        worlds_dir, 'feature_maps', world_setup + '.yaml')
    map_features_detection = os.path.join(
        worlds_dir, 'feature_maps', world_setup + '_detection' + '.yaml')
    rviz_config = os.path.join(localization_dir, 'rviz', 'pf_3d.rviz')

    # Webots
    webots = WebotsLauncher(world=world_file)

    # Controller
    robot_controller = WebotsController(
        robot_name='robot',
        parameters=[{'robot_description': robot_urdf}]
    )

    # Fake detector
    fake_detector = Node(
        package='robot_worlds',
        executable='fake_detector',
        name='fake_detector',
        output='screen',
        parameters=[{"map_features": map_features_detection},]
    )

    # 2D Ransac corner Detector
    corner_detector = Node(
        package="robot_feature_detector",
        executable="corners",
        name="corners",
        output="screen"
    )

    # Fake segmentator
    fake_segmentator = Node(
        package = "robot_pointnet",
        executable="fake_segmentator",
        name="fake_segmentator"
    )
    
    # PointNet segmentator
    segmentator = Node(
        package="robot_pointnet",
        executable="pointnet_segmentator",
        name="pointnet_segmentator"
    )

    # 3D Ransac corner detector
    corner_detector_3D = Node(
        package="robot_feature_detector",
        executable="corners_3D",
        name="corners_3D",
        output="screen"
    )

    # Particle filter
    particle_filter_config_file = os.path.join(
        get_package_share_directory('robot_localization_package'),
        'config',
        'particle_filter_params.yaml'
    )
    particle_filter = Node(
        package='robot_localization_package',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[
            particle_filter_config_file,
            {'map_features': map_features},
            {'map_yaml': map_yaml},
            {'map_pgm': map_pgm}]
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_yaml}],
        output='screen'
    )
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        parameters=[{'autostart': True, 'node_names': ['map_server']}],
        output='screen'
    )

    # TFs
    tf_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    tf_base_to_lidar_2D = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint_real', 'lidar2D']
    )
    tf_base_to_lidar_3D = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar_broadcaster',
        arguments=["0.13", "0", "0.25", "0", "0", "0", "base_footprint_real", "lidar3D"]
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # Teleop
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='gnome-terminal --'
    )

    # Path Tracker
    path_tracker = Node(
        package='robot_worlds',
        executable='path_tracker',
        name='path_tracker',
        output='screen'
    )

    return LaunchDescription([
        rviz,
        webots,
        robot_controller,
        particle_filter,
        #fake_detector,
        corner_detector_3D,
        #path_tracker,
        #fake_segmentator,
        segmentator,
        tf_map_to_odom,
        tf_base_to_lidar_2D,
        tf_base_to_lidar_3D,
        map_server,
        lifecycle_manager,
        teleop,
        RegisterEventHandler(
            OnProcessExit(
                target_action=webots,
                on_exit=[
                    launch.actions.EmitEvent(
                        event=Shutdown()
                    )
                ]
            )
        )
    ])
