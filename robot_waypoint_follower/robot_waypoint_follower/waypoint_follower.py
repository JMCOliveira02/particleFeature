import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
import math
import yaml
import os

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # Declare and read the YAML file path parameter
        self.declare_parameter('waypoints_file', '')
        waypoints_path = self.get_parameter('waypoints_file').get_parameter_value().string_value

        if not os.path.exists(waypoints_path):
            self.get_logger().error(f"Waypoint file not found: {waypoints_path}")
            rclpy.shutdown()
            return

        with open(waypoints_path, 'r') as f:
            data = yaml.safe_load(f)
            self.waypoints = data.get('waypoints', [])

        if not self.waypoints:
            self.get_logger().error("No waypoints loaded.")
            rclpy.shutdown()
            return

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.current_waypoint_idx = 0
        self.linear_speed = 0.5
        self.angular_gain = 1.5
        self.reach_threshold = 0.1

        self.get_logger().info('Waypoint Follower Initialized.')

    def control_loop(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint_real', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'Waiting for TF: {e}')
            return

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        # Convert quaternion to yaw
        q = tf.transform.rotation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))

        goal = self.waypoints[self.current_waypoint_idx]
        dx = goal[0] - x
        dy = goal[1] - y
        distance = math.hypot(dx, dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(angle_to_goal - yaw)

        if distance < self.reach_threshold:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx}")
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.get_logger().info("All waypoints reached.")
                self.cmd_pub.publish(Twist())
                self.destroy_timer(self.timer)
            return

        cmd = Twist()
        cmd.linear.x = self.linear_speed
        cmd.angular.z = self.angular_gain * angle_diff
        self.cmd_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main():
    rclpy.init()
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
