#include "robot_worlds/PathTracker.hpp"

PathTracker::PathTracker() : Node("path_tracker"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    RCLCPP_INFO(this->get_logger(), "PathTracker Constructor!");

    real_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("real_path", 10);
    estimated_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("estimated_path", 10);

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "estimated_pose", 10,
        std::bind(&PathTracker::poseCallback, this, std::placeholders::_1)
    );

    real_path.header.frame_id = "map";
    estimated_path.header.frame_id = "map";

    update_path_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&PathTracker::updatePaths, this)
    );
    
    // CSV file initialization
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::stringstream filename;
    std::string trial_type;


    this->declare_parameter("trial_type", std::string(""));
    this->get_parameter("trial_type", trial_type);

    filename << "/home/joao/ros2_ws/src/robot_worlds/trials/" << trial_type << "_";

    filename << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d_%H-%M") << ".csv";

    csv_file_.open(filename.str());
    csv_file_ << std::fixed << std::setprecision(3);

    if (!csv_file_.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file for writing.");
    } else {
        csv_file_ << "timestamp_real, timestamp_est, real_x, real_y, real_theta, est_x, est_y, est_theta, cov_x, cov_y, cov_theta\n";
    }

}

void PathTracker::updatePaths() {
    updatePathForFrame("base_footprint_real", real_path, real_path_pub_, true);
    updatePathForFrame("estimated_pose", estimated_path, estimated_path_pub_, false);
}


void PathTracker::updatePathForFrame(const std::string& frame_id, nav_msgs::msg::Path& path,
                            rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub, bool save_pose)
{
    geometry_msgs::msg::TransformStamped tf;
    try {
        tf = tf_buffer_.lookupTransform("map", frame_id, tf2::TimePointZero);
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "tf [%s] is not available!, error: %s", frame_id.c_str(), ex.what());
        return;
    }



    geometry_msgs::msg::PoseStamped pose;
    pose.header = tf.header;
    pose.pose.position.x = tf.transform.translation.x;
    pose.pose.position.y = tf.transform.translation.y;
    pose.pose.position.z = tf.transform.translation.z;
    pose.pose.orientation = tf.transform.rotation;
    if(save_pose){
        last_pose_ = pose;
    }
    path.header.stamp = this->get_clock()->now();
    path.poses.push_back(pose);
    pub->publish(path);
}

void PathTracker::poseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr est_pose) {
    if(last_pose_.header.stamp.sec == 0){
        RCLCPP_WARN(this->get_logger(), "No last pose available to update estimated path.");
        return;
    }
    
    double real_time = last_pose_.header.stamp.sec + last_pose_.header.stamp.nanosec * 1e-9;
    double est_time = est_pose->header.stamp.sec + est_pose->header.stamp.nanosec * 1e-9;
    
    double real_theta = tf2::getYaw(last_pose_.pose.orientation);
    double est_theta = tf2::getYaw(est_pose->pose.pose.orientation);
    
    if(csv_file_.is_open()){
        csv_file_ << real_time << "," << est_time << ","
        << last_pose_.pose.position.x << "," << last_pose_.pose.position.y << "," << real_theta << ","
        << est_pose->pose.pose.position.x << "," << est_pose->pose.pose.position.y << "," << est_theta << ","
        << est_pose->pose.covariance[0] << "," << est_pose->pose.covariance[7] << "," << est_pose->pose.covariance[35] << "\n";

        csv_file_.flush();
    }

}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathTracker>());
    rclcpp::shutdown();
    return 0;
}