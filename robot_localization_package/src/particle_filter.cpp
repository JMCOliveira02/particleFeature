#include "robot_localization_package/particle_filter.hpp"

struct PGMImage2
{
    int width;
    int height;
    int max_val;
    std::vector<uint8_t> data;

    void load(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open PGM file");

        std::string line;
        std::getline(file, line);
        if (line != "P5")
            throw std::runtime_error("Only binary PGM (P5) supported");

        do
        {
            std::getline(file, line);
        } while (line[0] == '#');

        std::stringstream ss(line);
        ss >> width >> height;

        file >> max_val;
        file.get(); // consume the newline

        data.resize(width * height);
        file.read(reinterpret_cast<char *>(data.data()), data.size());
    }

    uint8_t pixel(int x, int y) const
    {
        return data[y * width + x];
    }
};

std::tuple<double, double> pixelToWorld(int x_pix, int y_pix, double resolution, const std::vector<double> &origin, int image_height)
{
    double x = origin[0] + (x_pix + 0.5) * resolution;
    double y = origin[1] + (image_height - y_pix - 0.5) * resolution;
    return {x, y};
}

std::tuple<int, int> worldToPixel(double x_world, double y_world, double resolution, const std::vector<double> &origin, int image_height)
{
    int x_pix = static_cast<int>((x_world - origin[0]) / resolution - 0.5);
    int y_pix = image_height - 1 - static_cast<int>((y_world - origin[1]) / resolution - 0.5);
    return {x_pix, y_pix};
}

// Check if particle is in white part of pgm
bool isParticleInFreeSpace(double x_world, double y_world, const PGMImage &pgm, double resolution, const std::vector<double> &origin)
{
    // Convert world to pixel coordinates
    int x_pix = static_cast<int>((x_world - origin[0]) / resolution);
    int y_pix = pgm.height - 1 - static_cast<int>((y_world - origin[1]) / resolution); // y inverted

    // Check bounds
    if (x_pix < 0 || x_pix >= pgm.width || y_pix < 0 || y_pix >= pgm.height)
    {
        return false; // out of bounds = not free
    }

    uint8_t val = pgm.pixel(x_pix, y_pix);

    // Interpret pixel value
    return val >= 254; // white = free
}

ParticleFilter::ParticleFilter() : Node("particle_filter"),
                                    iterationCounter(0.0),
                                    msg_odom_base_link_(nullptr), 
                                    last_map_msg_(nullptr),
                                    first_odom_received_(false),      // Add these
                                    first_features_received_(false),   // Add these
                                    resample_cooldown_counter_(0)     // Initialize cooldown counter
{
    std::cout << "ParticleFilter Constructor START" << std::endl;
    RCLCPP_INFO(this->get_logger(), "Initializing particle filter node.");

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());

    // Initialize the listener to populate the buffer
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Load parameters from the parameter file
    loadParameters();

    // Load PGM and calculate free space
    calculateFreeSpaceFromPGM();

    distr_theta = std::uniform_real_distribution<double>(-M_PI, M_PI);
    distr_pgm_index = std::uniform_int_distribution<>(0, free_pixels.size() - 1);
    generator_ = std::mt19937(rd());

    // Retrieve the map_features parameter passed from the launch file

    this->get_parameter("map_features", map_features_);

    if (map_features_.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No map features provided. Please set the 'map_features' parameter.");
        return;
    }

    // Load the map features from the YAML file and store them in the global map
    map_loader_.loadToGlobalMap(map_features_);
    global_features_ = map_loader_.getGlobalFeatureMap();

    // Subscribe to odometry FIRST
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&ParticleFilter::motionUpdate, this, std::placeholders::_1));

    // Subscribe to features
    feature_sub_ = this->create_subscription<robot_msgs::msg::FeatureArray>(
        "/features", 10,
        std::bind(&ParticleFilter::storeMapMessage, this, std::placeholders::_1));

    // Create publishers for the estimated pose and particles
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/estimated_pose", 10);
    particles_color_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/particles_color", 10);
    particles_no_color_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/particles_no_color", 10);
    feature_map_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/feature_map_markers", 10);

    // Create a transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Create a timer to publish the estimated pose (computed in motion/measurement updates)
    timer_pose_ = create_wall_timer(std::chrono::milliseconds(200), std::bind(&ParticleFilter::publishEstimatedPose, this));
    
    // Create a timer to publish the feature map markers for visualization
    timer_feature_map_ = create_wall_timer(std::chrono::milliseconds(2000), std::bind(&ParticleFilter::publishFeatureMapMarkers, this));

    // Initialize the color pallete for the particles weights
    computeColorWeightLookup();

    init_weight = 1.0 / num_particles_;

    // Initialize the particles
    initializeParticles_pgm();

    RCLCPP_INFO(this->get_logger(), "Particle filter node initialized successfully.");
}

//! auxiliar functions start!//

#pragma region auxiliar functions

// load the paramaters for the node
void ParticleFilter::loadParameters()
{
    RCLCPP_INFO(this->get_logger(), "Loading particle filter parameters...");

    bool success = true;

    this->declare_parameter("num_particles", NUM_PARTICLES);
    this->declare_parameter("motion_delta_distance", MOTION_DELTA_DISTANCE);
    this->declare_parameter("motion_delta_angle", MOTION_DELTA_ANGLE);
    this->declare_parameter("motion_x_variance", MOTION_X_VARIANCE);
    this->declare_parameter("motion_y_variance", MOTION_Y_VARIANCE);
    this->declare_parameter("motion_angle_variance", MOTION_ANGLE_VARIANCE);
    this->declare_parameter("resample_ess_threshold", RESAMPLE_ESS_THRESHOLD);
    this->declare_parameter("resample_max_weight_threshold", RESAMPLE_MAX_WEIGHT_THRESHOLD);
    this->declare_parameter("inject_num_iterations", INJECT_NUM_ITERATIONS);
    this->declare_parameter("inject_percentage", INJECT_PERCENTAGE);
    this->declare_parameter("replace_worst_percentage", REPLACE_WORST_PERCENTAGE);
    this->declare_parameter("estimate_num_particles", ESTIMATE_NUM_PARTICLES);
    this->declare_parameter("map_features", std::string(""));
    this->declare_parameter("map_yaml", std::string(""));
    this->declare_parameter("map_pgm", std::string(""));

    success &= this->get_parameter("num_particles", num_particles_);
    success &= this->get_parameter("motion_delta_distance", motion_delta_distance_);
    success &= this->get_parameter("motion_delta_angle", motion_delta_angle_);
    success &= this->get_parameter("motion_x_variance", motion_x_variance_);
    success &= this->get_parameter("motion_y_variance", motion_y_variance_);
    success &= this->get_parameter("motion_angle_variance", motion_angle_variance_);
    success &= this->get_parameter("resample_ess_threshold", resample_ess_threshold_);
    success &= this->get_parameter("resample_max_weight_threshold", resample_max_weight_threshold_);
    success &= this->get_parameter("inject_num_iterations", inject_num_iterations_);
    success &= this->get_parameter("inject_percentage", inject_percentage_);
    success &= this->get_parameter("replace_worst_percentage", replace_worst_percentage_);
    success &= this->get_parameter("estimate_num_particles", estimate_num_particles_);
    success &= this->get_parameter("map_features", map_features_);
    success &= this->get_parameter("map_yaml", map_yaml_);
    success &= this->get_parameter("map_pgm", map_pgm_);

    if (!success)
    {
        RCLCPP_ERROR(this->get_logger(), "One or more parameters failed to load. Check your YAML or launch file.");
        // Optionally shutdown or throw an exception here
        return;
    }

    // Log loaded parameters
    RCLCPP_INFO(this->get_logger(), "num_particles: %.1f", num_particles_);
    RCLCPP_INFO(this->get_logger(), "motion_delta_distance: %.2f", motion_delta_distance_);
    RCLCPP_INFO(this->get_logger(), "motion_delta_angle: %.2f", motion_delta_angle_);
    RCLCPP_INFO(this->get_logger(), "motion_x_variance: %.2f", motion_x_variance_);
    RCLCPP_INFO(this->get_logger(), "motion_y_variance: %.2f", motion_y_variance_);
    RCLCPP_INFO(this->get_logger(), "motion_angle_variance: %.2f", motion_angle_variance_);
    RCLCPP_INFO(this->get_logger(), "resample_ess_threshold: %.2f", resample_ess_threshold_);
    RCLCPP_INFO(this->get_logger(), "resample_max_weight_threshold: %.2f", resample_max_weight_threshold_);
    RCLCPP_INFO(this->get_logger(), "inject_num_iterations: %d", inject_num_iterations_);
    RCLCPP_INFO(this->get_logger(), "inject_percentage: %.2f", inject_percentage_);
    RCLCPP_INFO(this->get_logger(), "replace_worst_percentage: %.2f", replace_worst_percentage_);
    RCLCPP_INFO(this->get_logger(), "estimate_num_particles: %d", estimate_num_particles_);
    RCLCPP_INFO(this->get_logger(), "map_features: %s", map_features_.c_str());
    RCLCPP_INFO(this->get_logger(), "map_yaml: %s", map_yaml_.c_str());
    RCLCPP_INFO(this->get_logger(), "map_pgm: %s", map_pgm_.c_str());
}


void ParticleFilter::calculateFreeSpaceFromPGM()
{
    // Load map.yaml
    YAML::Node config = YAML::LoadFile(map_yaml_);
    resolution = config["resolution"].as<double>();
    origin = config["origin"].as<std::vector<double>>();
    int negate = config["negate"] ? config["negate"].as<int>() : 0;

    // Load PGM
    // PGMImage pgm;
    pgm.load(map_pgm_);

    // Collect free pixels
    // std::vector<std::pair<int, int>> free_pixels;
    for (int y = 0; y < pgm.height; ++y)
    {
        for (int x = 0; x < pgm.width; ++x)
        {
            uint8_t val = pgm.pixel(x, y);
            bool is_free = (negate == 0) ? (val >= 254) : (val <= 1);
            if (is_free)
            {
                free_pixels.emplace_back(x, y);
            }
        }
    }
}

// normalize the weights of the particles
void ParticleFilter::normalizeWeights()
{
    double sum_weights = std::accumulate(particles_.begin(), particles_.end(), 0.0,
                                         [](double sum, const Particle &p)
                                         { return sum + p.weight; });

    for (auto &p : particles_)
    {
        p.weight /= sum_weights;
    }
}

// get the maximum weight of the particles
double ParticleFilter::maxWeight()
{
    double max_weight = 0.0;
    for (const auto &p : particles_)
    {
        max_weight = std::max(max_weight, p.weight);
    }
    return max_weight;
}

// compute the color weight lookup table for visualization
void ParticleFilter::computeColorWeightLookup()
{
    double average_weight = 1.0 / num_particles_;

    ColorWeightLookup = {
        {0.9 * average_weight, {1.0, 1.0, 1.0}},              // White
        {average_weight, {0.56, 0.0, 1.0}},                   // Violet
        {1.05 * average_weight, {0.0, 0.0, 1.0}},             // Blue
        {1.1 * average_weight, {0.0, 1.0, 0.5}},              // Cyan
        {1.15 * average_weight, {0.0, 1.0, 0.0}},             // Green
        {1.2 * average_weight, {1.0, 1.0, 0.0}},              // Yellow
        {1.5 * average_weight, {1.0, 0.5, 0.0}},              // Orange
        {std::numeric_limits<double>::max(), {1.0, 0.0, 0.0}} // Red
    };
}

// get color based on the weight of the particle
std::vector<double> ParticleFilter::colorFromWeight(double weight) const
{
    for (const auto &entry : ColorWeightLookup)
    {
        if (weight < entry.first)
        {
            return entry.second;
        }
    }
    RCLCPP_WARN(this->get_logger(), "Weight out of range: %f", weight);
    return {1.0, 0.0, 0.0}; // Default to Red
}

// publish the particles for visualization as markers
void ParticleFilter::publishParticles_with_color()
{
    if (particles_.empty())
        return;

    visualization_msgs::msg::MarkerArray marker_array;
    int i = 0;

    for (const auto &p : particles_)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "particle";
        marker.id = i++;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = p.x;
        marker.pose.position.y = p.y;
        marker.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, p.theta);
        marker.pose.orientation = tf2::toMsg(q);

        marker.scale.x = 0.07;
        marker.scale.y = 0.005;
        marker.scale.z = 0.01;

        auto color = colorFromWeight(p.weight);
        marker.color.a = 1.0;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];

        marker_array.markers.push_back(marker);
    }

    particles_color_pub_->publish(marker_array);
}

void ParticleFilter::publishParticles_no_color()
{
    if (particles_.empty())
        return;

    geometry_msgs::msg::PoseArray pose_array;

    pose_array.header.frame_id = "map";
    pose_array.header.stamp = this->get_clock()->now();
    
    for (const auto &p : particles_)
    {
        geometry_msgs::msg::Pose pose;
        pose.position.x = p.x;
        pose.position.y = p.y;
        pose.position.z = 0.0;
    
        tf2::Quaternion q;
        q.setRPY(0, 0, p.theta);
        pose.orientation = tf2::toMsg(q);
    
        pose_array.poses.push_back(pose);
    }
    particles_no_color_pub_->publish(pose_array);
}

void ParticleFilter::publishFeatureMapMarkers()
{
    if (global_features_.empty())
        return;

    visualization_msgs::msg::MarkerArray marker_array;

    // Clear existing markers first
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "map";
    clear_marker.header.stamp = this->get_clock()->now();
    clear_marker.ns = "feature_map";
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    // Add markers for each feature in the global map
    int marker_id = 0;
    for (const auto& feature_ptr : global_features_)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "feature_map";
        marker.id = marker_id++;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.lifetime = rclcpp::Duration(0, 0); // Never expire

        // Set position
        marker.pose.position.x = feature_ptr->x;
        marker.pose.position.y = feature_ptr->y;
        marker.pose.position.z = 0.05; // Raise it a bit above ground for visibility

        // Set orientation - convert from degrees to radians
        tf2::Quaternion q;
        // Most features have meaningful orientation, only circular columns don't
        bool has_angle = (feature_ptr->type != "column_circular");
        if (has_angle)
        {
            // Convert degrees to radians for tf2
            double theta_radians = feature_ptr->theta * M_PI / 180.0;
            q.setRPY(0, 0, theta_radians);
            RCLCPP_DEBUG(this->get_logger(), "Feature %s at (%.2f, %.2f) with angle %.1f° (%.3f rad)", 
                        feature_ptr->type.c_str(), feature_ptr->x, feature_ptr->y, 
                        feature_ptr->theta, theta_radians);
        }
        else
        {
            q.setRPY(0, 0, 0);
            RCLCPP_DEBUG(this->get_logger(), "Feature %s at (%.2f, %.2f) without angle (circular column)", 
                        feature_ptr->type.c_str(), feature_ptr->x, feature_ptr->y);
        }
        marker.pose.orientation = tf2::toMsg(q);

        // Set marker type and appearance based on feature type - all as arrows
        marker.type = visualization_msgs::msg::Marker::ARROW;
        
        if (feature_ptr->type == "column" || feature_ptr->type == "column_circular")
        {
            // Green arrows for columns
            marker.scale.x = 0.8;  // length
            marker.scale.y = 0.15; // width
            marker.scale.z = 0.15; // height
            
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.8;
        }
        else if (feature_ptr->type == "window" || feature_ptr->type == "window_big" || feature_ptr->type == "window_small")
        {
            // Bright cyan arrows for windows - longer to show orientation clearly, maximum visibility on dark backgrounds
            marker.scale.x = 1.2;  // length
            marker.scale.y = 0.15;  // width
            marker.scale.z = 0.15;  // height
            
            marker.color.r = 0.0;  // Bright cyan
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;  // Full opacity for maximum brightness
        }
        else if (feature_ptr->type == "chair")
        {
            // Purple arrows for chairs
            marker.scale.x = 0.6;  // length
            marker.scale.y = 0.16; // width
            marker.scale.z = 0.08; // height
            
            marker.color.r = 0.5;
            marker.color.g = 0.0;
            marker.color.b = 0.5;
            marker.color.a = 0.8;
        }
        else if (feature_ptr->type == "table")
        {
            // Orange arrows for tables
            marker.scale.x = 1.0;  // length
            marker.scale.y = 0.12; // width
            marker.scale.z = 0.12; // height
            
            marker.color.r = 1.0;
            marker.color.g = 0.5;
            marker.color.b = 0.0;
            marker.color.a = 0.8;
        }
        else
        {
            // Red arrows for unknown types
            marker.scale.x = 0.6;  // length
            marker.scale.y = 0.1;  // width
            marker.scale.z = 0.1;  // height
            
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.8;
        }

        // Add text label above the marker
        marker_array.markers.push_back(marker);

        // Create text marker for label
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = "map";
        text_marker.header.stamp = this->get_clock()->now();
        text_marker.ns = "feature_map_labels";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.lifetime = rclcpp::Duration(0, 0);

        text_marker.pose.position.x = feature_ptr->x;
        text_marker.pose.position.y = feature_ptr->y;
        text_marker.pose.position.z = 1.5; // Above the main marker

        text_marker.scale.z = 0.3; // Text size
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;

        std::stringstream label_stream;
        label_stream << feature_ptr->type;
        if (has_angle)
        {
            label_stream << " (" << std::fixed << std::setprecision(0) 
                        << feature_ptr->theta << "°)";  // Show original degrees value
        }
        text_marker.text = label_stream.str();

        marker_array.markers.push_back(text_marker);
    }

    feature_map_pub_->publish(marker_array);
    
    RCLCPP_DEBUG(this->get_logger(), "Published %zu feature map markers", global_features_.size());
}

// replace the worst particles with random ones in white part of pgm (free space)
void ParticleFilter::replaceWorstParticles_pgm(double percentage)
{
    std::sort(particles_.begin(), particles_.end(),
              [](const Particle &a, const Particle &b)
              { return a.weight < b.weight; });

    int num_replace = static_cast<int>(num_particles_ * percentage);

    for (int i = 0; i < num_replace; i++)
    {
        auto [x_pix, y_pix] = free_pixels[distr_pgm_index(generator_)];
        auto [x, y] = pixelToWorld(x_pix, y_pix, resolution, origin, pgm.height);

        particles_[i].x = x;
        particles_[i].y = y;
        particles_[i].theta = distr_theta(generator_);
        particles_[i].weight = init_weight;
    }

    normalizeWeights();
}

// replace and inject random particles into the filter in white part of pgm (free space)
void ParticleFilter::injectRandomParticles_pgm(double percentage)
{

    int num_replace = static_cast<int>(num_particles_ * percentage);

    std::vector<int> random_indices(num_replace);
    for (int &index : random_indices)
    {
        index = rand() % static_cast<int>(num_particles_);
    }

    for (int i : random_indices)
    {
        auto [x_pix, y_pix] = free_pixels[distr_pgm_index(generator_)];
        auto [x, y] = pixelToWorld(x_pix, y_pix, resolution, origin, pgm.height);

        particles_[i].x = x;
        particles_[i].y = y;
        particles_[i].theta = distr_theta(generator_);
        particles_[i].weight = init_weight;
    }
}

#pragma endregion auxiliar functions

//! auxiliar functions end!//

//! Feature Handling start !//

#pragma region feature handling

void ParticleFilter::storeMapMessage(const robot_msgs::msg::FeatureArray::SharedPtr msg)
{    
    if (!first_odom_received_) {
        RCLCPP_WARN(this->get_logger(), "Received features before odometry - skipping");
        return;
    }
    // Received Features
    last_map_msg_ = msg;

    /* // Calculate motion since last measurement update
    double delta_x = odom_x - last_update_x_;
    double delta_y = odom_y - last_update_y_;
    double delta_theta = std::abs(odom_theta - last_update_theta_);
    double delta_distance = std::hypot(delta_x, delta_y);
    
    RCLCPP_DEBUG(this->get_logger(), "Motion since last update: distance=%.3f, angle=%.3f", 
                 delta_distance, delta_theta);
                 
    // Only do measurement update if we've moved enough
    if (delta_distance > motion_delta_distance_ || delta_theta > motion_delta_angle_)
    { */
    //RCLCPP_INFO(this->get_logger(), "🔄 Measurement update! Moved %.3fm, rotated %.3f rad since last update", 
    //            delta_distance, delta_theta);
    RCLCPP_INFO(this->get_logger(), "   Processing %zu features", msg->features.size());
    
    // Perform measurement update
    measurementUpdate(last_map_msg_);
    
    /* // Update the last positions where we did a measurement
    last_update_x_ = odom_x;
    last_update_y_ = odom_y;
    last_update_theta_ = odom_theta; */
    /* } else {
        RCLCPP_DEBUG(this->get_logger(), "Not enough motion for measurement update (%.3fm < %.3fm)", 
                     delta_distance, motion_delta_distance_);
    } */

}

std::vector<map_features::Feature> ParticleFilter::getExpectedFeatures(const Particle &p, double delta_scan_x, double delta_scan_y, double delta_scan_theta, const std::string &type)
{
    std::vector<map_features::Feature> features_particle;

    double scan_p_x = p.x - delta_scan_x;
    double scan_p_y = p.y - delta_scan_y;
    double scan_p_theta = p.theta - delta_scan_theta;
    if (scan_p_theta < -M_PI)
        scan_p_theta += 2 * M_PI;
    else if (scan_p_theta > M_PI)
        scan_p_theta -= 2 * M_PI;



    double cos_theta = std::cos(scan_p_theta);
    double sin_theta = std::sin(scan_p_theta);

    for (const auto &feature_ptr : global_features_)
    {
        if (feature_ptr->type == type)
        {
            auto object_ptr = std::dynamic_pointer_cast<map_features::Feature>(feature_ptr);
            if (!object_ptr)
                continue;

            double map_x = object_ptr->x;
            double map_y = object_ptr->y;
            double feature_theta = object_ptr->theta * M_PI / 180.0;  // Convert degrees to radians

            double particle_x = cos_theta * (map_x - scan_p_x) + sin_theta * (map_y - scan_p_y);
            double particle_y = -sin_theta * (map_x - scan_p_x) + cos_theta * (map_y - scan_p_y);

            features_particle.emplace_back(particle_x, particle_y, feature_theta, type);
        }
    }

    return features_particle;
}

// transform angle from the map frame to the particle frame
double ParticleFilter::transformAngleToParticleFrame(double feature_theta_map, double particle_theta)
{
    // DEBUG: Log input angles
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: transformAngleToParticleFrame - Input: feature_theta_map=%.3f rad (%.1f°), particle_theta=%.3f rad (%.1f°)", 
                feature_theta_map, feature_theta_map * 180.0 / M_PI, particle_theta, particle_theta * 180.0 / M_PI);

    // Normalize both angles to [-π, π]
    auto normalize_angle = [](double angle) {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    };

    feature_theta_map = normalize_angle(feature_theta_map);
    particle_theta = normalize_angle(particle_theta);

    // ✅ FIXED: Proper coordinate frame transformation for angles
    // Transform the feature's orientation from world frame to particle's local frame
    // This is equivalent to rotating the feature's orientation vector by -particle_theta
    double feature_in_particle_frame = normalize_angle(feature_theta_map - particle_theta);

    // DEBUG: Log transformation steps
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: Normalized inputs: feature_map=%.3f rad (%.1f°), particle=%.3f rad (%.1f°)", 
                feature_theta_map, feature_theta_map * 180.0 / M_PI, particle_theta, particle_theta * 180.0 / M_PI);
    
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: Feature in particle frame: %.3f rad (%.1f°)", 
                feature_in_particle_frame, feature_in_particle_frame * 180.0 / M_PI);

    // The result represents how the feature is oriented relative to the particle's coordinate system
    // For example:
    // - 0°: Feature is aligned with particle's forward direction
    // - 90°: Feature is oriented 90° counterclockwise from particle's forward direction
    // - -90°: Feature is oriented 90° clockwise from particle's forward direction

    return feature_in_particle_frame;
}

// compute the likelihood for the orientation of the corner feature
double ParticleFilter::computeAngleLikelihood(double measured_angle, double expected_angle, double sigma)
{
    // DEBUG: Log input angles
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: computeAngleLikelihood - Input: measured_angle=%.3f rad (%.1f°), expected_angle=%.3f rad (%.1f°), sigma=%.3f", 
                measured_angle, measured_angle * 180.0 / M_PI, expected_angle, expected_angle * 180.0 / M_PI, sigma);

    if (measured_angle > M_PI)
        measured_angle -= 2 * M_PI;
    if (measured_angle < -M_PI)
        measured_angle += 2 * M_PI;
    if (expected_angle > M_PI)
        expected_angle -= 2 * M_PI;
    if (expected_angle < -M_PI)
        expected_angle += 2 * M_PI;

    double error = measured_angle - expected_angle;

    // DEBUG: Log initial error
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: Initial angle error: %.3f rad (%.1f°)", error, error * 180.0 / M_PI);

    while (error > M_PI)
        error -= 2 * M_PI;
    while (error < -M_PI)
        error += 2 * M_PI;

    // DEBUG: Log normalized error
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: Normalized angle error: %.3f rad (%.1f°)", error, error * 180.0 / M_PI);

    // ✅ FIXED: Correct Gaussian formula - removed extra factor of 2 in denominator
    // Standard Gaussian: exp(-0.5 * error^2 / sigma^2)
    double exponent = -0.5 * (error * error) / (sigma * sigma);
    double likelihood = std::exp(exponent);

    // DEBUG: Log likelihood computation
    RCLCPP_DEBUG(this->get_logger(), "ANGLE_DEBUG: Likelihood computation: exponent=%.3f, likelihood=%.6f", exponent, likelihood);

    return likelihood;
}

// compute the likelihood of a corner feature based on distance and angle
double ParticleFilter::computeLikelihoodFeature(const Particle &p, double delta_scan_x, double delta_scan_y, double delta_scan_theta, double noisy_x, double noisy_y, double measured_theta, double sigma_pos, double sigma_theta, const std::string &type)
{
    std::vector<map_features::Feature> expected_features = getExpectedFeatures(p,delta_scan_x, delta_scan_y, delta_scan_theta, type);

    // DEBUG: Log particle and feature details
    RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: Particle at (%.3f, %.3f, %.1f°), feature type='%s', measured at (%.3f, %.3f), with_angle=%s, measured_theta=%.3f rad (%.1f°)", 
                p.x, p.y, p.theta * 180.0 / M_PI, type.c_str(), noisy_x, noisy_y, 
                with_angle_ ? "true" : "false", measured_theta, measured_theta * 180.0 / M_PI);

    RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: Found %zu expected features for type '%s'", expected_features.size(), type.c_str());

    double min_dist = std::numeric_limits<double>::max();
    map_features::Feature best_feature(0, 0, 0, type);

    double likelihood = 0.0;

    for (size_t i = 0; i < expected_features.size(); i++)
    {
        const auto &exp = expected_features[i];
        double dist = std::hypot(noisy_x - exp.x, noisy_y - exp.y);
        
        RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: Expected feature %zu at (%.3f, %.3f), distance=%.3f", 
                    i, exp.x, exp.y, dist);
        
        if (dist < min_dist)
        {
            min_dist = dist;
            best_feature = exp;
            RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: New best feature at distance %.3f", min_dist);
        }
    }

    // Compute likelihood based on distance and angle
    double scan_p_theta = p.theta - delta_scan_theta;
    if (scan_p_theta < -M_PI)
        scan_p_theta += 2 * M_PI;
    else if (scan_p_theta > M_PI)
        scan_p_theta -= 2 * M_PI;

    RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: scan_p_theta=%.3f rad (%.1f°), best_feature.theta=%.3f rad (%.1f°)", 
                scan_p_theta, scan_p_theta * 180.0 / M_PI, best_feature.theta, best_feature.theta * 180.0 / M_PI);

    double expected_feature_angle = transformAngleToParticleFrame(best_feature.theta, scan_p_theta);
    double angle_likelihood = computeAngleLikelihood(measured_theta, expected_feature_angle, sigma_theta);
    // ✅ FIXED: Correct Gaussian formula for distance likelihood
    double distance_likelihood = std::exp(-(min_dist * min_dist) / (2.0 * sigma_pos * sigma_pos));
    
    RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: min_dist=%.3f, distance_likelihood=%.6f, sigma_pos=%.3f, expected_angle=%.3f rad (%.1f°)", 
                min_dist, distance_likelihood, sigma_pos, expected_feature_angle, expected_feature_angle * 180.0 / M_PI);

    if (with_angle_)
    {
        // ✅ FIXED: Use standard likelihood combination - no artificial multipliers or powers
        // Combined likelihood as simple product (assuming independence)
        likelihood = angle_likelihood * distance_likelihood;
        RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: WITH angle - angle_likelihood=%.6f, distance_likelihood=%.6f, combined=%.6f", 
                    angle_likelihood, distance_likelihood, likelihood);
    }
    else
    {
        // ✅ FIXED: Use standard distance likelihood - no artificial multipliers or powers
        likelihood = distance_likelihood;
        RCLCPP_DEBUG(this->get_logger(), "LIKELIHOOD_DEBUG: WITHOUT angle - distance_likelihood=%.6f", likelihood);
    }

    return likelihood;
}

// decode a feature message received from the topic features into a DecodedMsg structure
ParticleFilter::DecodedMsg ParticleFilter::decodeMsg(const robot_msgs::msg::Feature &msg)
{
    DecodedMsg feature;

    feature.x = msg.x;
    feature.y = msg.y;
    feature.theta = msg.theta;
    feature.type = msg.type;
    feature.with_angle = msg.with_angle;
    feature.confidence = msg.confidence;
    feature.angle_variance = msg.orientation_variance;

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            feature.covariance_pos[i][j] = msg.position_covariance[i * 2 + j];
        }
    }

    return feature;
}

#pragma endregion feature handling

//! Feature Handling end !//

//! Resampling functions start !//

#pragma region resampling functions

void ParticleFilter::multinomialResample()
{
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    std::vector<double> cumulative_weights(num_particles_);
    cumulative_weights[0] = particles_[0].weight;
    for (size_t i = 1; i < num_particles_; i++)
    {
        cumulative_weights[i] = cumulative_weights[i - 1] + particles_[i].weight;
    }

    std::uniform_real_distribution<double> dist(0.0, cumulative_weights.back());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);

    for (size_t i = 0; i < num_particles_; i++)
    {
        double r = dist(generator_);
        auto it = std::lower_bound(cumulative_weights.begin(), cumulative_weights.end(), r);
        int index = std::distance(cumulative_weights.begin(), it);
        new_particles.push_back(particles_[index]);
    }

    particles_ = new_particles;
}

void ParticleFilter::stratifiedResample()
{
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    std::vector<double> cumulative_weights(num_particles_);
    cumulative_weights[0] = particles_[0].weight;
    for (size_t i = 1; i < num_particles_; i++)
    {
        cumulative_weights[i] = cumulative_weights[i - 1] + particles_[i].weight;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);

    int index = 0;
    for (size_t i = 0; i < num_particles_; i++)
    {
        double r = dist(generator_); // Generate new random variable for each particle
        double U = r + (i / static_cast<double>(num_particles_));
        while (U > cumulative_weights[index])
            index++;
        new_particles.push_back(particles_[index]);
    }

    particles_ = new_particles;
}

void ParticleFilter::systematicResample()
{
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    // Compute cumulative weights
    std::vector<double> cumulative_weights(num_particles_);
    cumulative_weights[0] = particles_[0].weight;
    for (size_t i = 1; i < num_particles_; i++)
    {
        cumulative_weights[i] = cumulative_weights[i - 1] + particles_[i].weight;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0 / num_particles_);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_.seed(seed);
    double r = dist(generator_);
    int index = 0;
    for (size_t i = 0; i < num_particles_; i++)
    {
        double U = r + (i / static_cast<double>(num_particles_));
        while (U > cumulative_weights[index])
            index++;
        new_particles.push_back(particles_[index]);
    }

    particles_ = new_particles;
}

void ParticleFilter::residualResample()
{
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    std::vector<double> residual_weights;
    int total_copies = 0;

    for (const auto &p : particles_)
    {
        int num_copies = static_cast<int>(p.weight * num_particles_);
        total_copies += num_copies;
        for (int j = 0; j < num_copies; j++)
            new_particles.push_back(p);
        residual_weights.push_back((p.weight * num_particles_) - num_copies);
    }

    std::vector<double> cumulative_weights;
    double sum_residuals = 0.0;
    for (double rw : residual_weights)
    {
        sum_residuals += rw;
        cumulative_weights.push_back(sum_residuals);
    }

    std::uniform_real_distribution<double> dist(0.0, sum_residuals);

    while (total_copies < num_particles_)
    {
        double r = dist(generator_);
        for (size_t i = 0; i < cumulative_weights.size(); i++)
        {
            if (r <= cumulative_weights[i])
            {
                new_particles.push_back(particles_[i]);
                total_copies++;
                break;
            }
        }
    }

    particles_ = new_particles;
}

#pragma endregion resampling functions

//! Resampling functions end !//

//! Particle Filter Functions !//

#pragma region pf functions

void ParticleFilter::initializeParticle(Particle &p, double weight)
{
    auto [x_pix, y_pix] = free_pixels[distr_pgm_index(generator_)];
    auto [x, y] = pixelToWorld(x_pix, y_pix, resolution, origin, pgm.height);
    p.x = x;
    p.y = y;
    p.theta = distr_theta(generator_);
    p.weight = weight;
}

void ParticleFilter::initializeParticles_pgm()
{
    RCLCPP_INFO(this->get_logger(),"init weight: %.2f", init_weight );
    particles_.resize(num_particles_);
    for (auto &p : particles_)
    {
        initializeParticle(p, init_weight);
    }
}

void ParticleFilter::motionUpdate(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (particles_.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No particles to update.");
        return;
    }

    msg_odom_base_link_ = msg;

    odom_x = msg->pose.pose.position.x;
    odom_y = msg->pose.pose.position.y;

    tf2::Quaternion odom_q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);

    double roll, pitch;
    tf2::Matrix3x3(odom_q).getRPY(roll, pitch, odom_theta);

    // Handle first odometry message
    if (!first_odom_received_)
    {
        RCLCPP_INFO(this->get_logger(), "🚗 First odometry received at position (%.3f, %.3f, %.3f rad)", 
                    odom_x, odom_y, odom_theta);
        
        // Initialize all tracking variables
        last_odom_x_ = odom_x;
        last_odom_y_ = odom_y;
        last_odom_theta_ = odom_theta;
        
        // IMPORTANT: Also initialize the last_update positions
        // This prevents huge jumps when first feature arrives
        last_update_x_ = odom_x;
        last_update_y_ = odom_y;
        last_update_theta_ = odom_theta;
        
        // Initialize estimated pose to current odometry
        x_last_final = odom_x;
        y_last_final = odom_y;
        theta_last_final = odom_theta;
        
        first_odom_received_ = true;
        return; // Skip motion update on first message
    }

    double delta_x_odom = odom_x - last_odom_x_;
    double delta_y_odom = odom_y - last_odom_y_;
    double delta_distance = std::hypot(delta_x_odom, delta_y_odom);

    std::uniform_real_distribution<double> noise_x(-motion_x_variance_, motion_x_variance_);
    std::uniform_real_distribution<double> noise_y(-motion_y_variance_, motion_y_variance_);
    std::uniform_real_distribution<double> noise_theta(-motion_angle_variance_, motion_angle_variance_);

    double alpha_odom = atan2(delta_y_odom, delta_x_odom);
    double alpha_robot = alpha_odom - last_odom_theta_;
    double delta_x_robot = delta_distance * std::cos(alpha_robot);
    double delta_y_robot = delta_distance * std::sin(alpha_robot);
    double delta_theta_odom = odom_theta - last_odom_theta_;

    // Normalize angle
    while (delta_theta_odom > M_PI) delta_theta_odom -= 2 * M_PI;
    while (delta_theta_odom < -M_PI) delta_theta_odom += 2 * M_PI;


    // update particles if significant motion is detected
    if (delta_distance > motion_delta_distance_ || std::abs(delta_theta_odom) > motion_delta_angle_)
    {
        RCLCPP_DEBUG(this->get_logger(), "🤖 Updating particles: moved %.3fm, rotated %.3f rad", 
                     delta_distance, delta_theta_odom);

        for (auto &p : particles_)
        {
            p.x += delta_x_robot * std::cos(p.theta) - delta_y_robot * std::sin(p.theta) + noise_x(generator_);
            p.y += delta_x_robot * std::sin(p.theta) + delta_y_robot * std::cos(p.theta) + noise_y(generator_);
            p.theta += delta_theta_odom + noise_theta(generator_);
            
            bool penalize = !isParticleInFreeSpace(p.x, p.y, pgm, resolution, origin);
            if (penalize)
            {
                p.weight *= 0.01; // LecomputeLikelihoodFeaturess harsh than /4
            } 

            if (p.theta > M_PI)
                p.theta -= 2 * M_PI;
            if (p.theta < -M_PI)
                p.theta += 2 * M_PI;
        }

        // 🎯 ALWAYS move the estimated pose with the same motion as particles
        double prev_pose_x = x_last_final;
        double prev_pose_y = y_last_final;
        double prev_pose_theta = theta_last_final;
        
        x_last_final += delta_x_robot * std::cos(theta_last_final) - delta_y_robot * std::sin(theta_last_final);
        y_last_final += delta_x_robot * std::sin(theta_last_final) + delta_y_robot * std::cos(theta_last_final);
        theta_last_final += delta_theta_odom;
        
        // Normalize estimated pose angle
        if (theta_last_final > M_PI)
            theta_last_final -= 2 * M_PI;
        if (theta_last_final < -M_PI)
            theta_last_final += 2 * M_PI;
        
        // Check for huge motion jumps (could indicate odometry issue)
        double pose_motion = std::hypot(x_last_final - prev_pose_x, y_last_final - prev_pose_y);
        if (pose_motion > 1.0) {  // More than 1 meter in one update
            RCLCPP_WARN(this->get_logger(), "🚨 LARGE MOTION DETECTED: %.3fm in one update - delta_robot=(%.3f,%.3f,%.2f°)", 
                        pose_motion, delta_x_robot, delta_y_robot, delta_theta_odom * 180.0 / M_PI);
        }
            
        RCLCPP_DEBUG(this->get_logger(), "🎯 Updated estimated pose with motion: (%.3f, %.3f, %.1f°)", 
                     x_last_final, y_last_final, theta_last_final * 180.0 / M_PI);

        last_odom_x_ = odom_x;
        last_odom_y_ = odom_y;
        last_odom_theta_ = odom_theta;

        // Compute updated pose after motion (but don't publish yet - timer will handle publishing)
        computeEstimatedPose();
    }

    if(with_color_){
        publishParticles_with_color();
    }   
    else{
        publishParticles_no_color();
    }
}
  
void ParticleFilter::measurementUpdate(const robot_msgs::msg::FeatureArray::SharedPtr msg)
{
    if (particles_.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No particles to update.");
        return;
    }

    // Handle empty feature messages gracefully
    if (msg->features.empty())
    {
        RCLCPP_INFO(this->get_logger(), "No features detected - maintaining particle weights");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "IM HEREEEE");


    rclcpp::Time scan_timestamp = msg->header.stamp;

    geometry_msgs::msg::TransformStamped scan_tf;

    double scan_pose_x, scan_pose_y, scan_pose_theta;

    bool all_outside = true;

    try {
        scan_tf = tf_buffer_->lookupTransform(
            "odom",              // target frame
            "base_footprint",    // source frame
            scan_timestamp,                   // timestamp
            rclcpp::Duration::from_seconds(0.1)  // timeout
        );

        // Convert to PoseStamped

        scan_pose_x = scan_tf.transform.translation.x;
        scan_pose_y = scan_tf.transform.translation.y;
        scan_pose_theta = tf2::getYaw(scan_tf.transform.rotation);


    // Now pose contains the pose of base_footprint in odom frame at time x
    } catch (const tf2::TransformException & ex) {
        scan_pose_x = odom_x;
        scan_pose_y = odom_y;
        scan_pose_theta = odom_theta;
        RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
    }

    double delta_scan_x, delta_scan_y, delta_scan_theta;
    delta_scan_x = odom_x - scan_pose_x;
    delta_scan_y = odom_y - scan_pose_y;
    delta_scan_theta = odom_theta - scan_pose_theta;
    if (delta_scan_theta < -M_PI)
        delta_scan_theta += 2 * M_PI;
    else if (delta_scan_theta > M_PI)
        delta_scan_theta -= 2 * M_PI;


    for (size_t particle_idx = 0; particle_idx < particles_.size(); particle_idx++)
    {
        auto &p = particles_[particle_idx];
        double likelihood = 0; // Start with 1.0

        for (const auto &obs_msg : msg->features)
        {

            DecodedMsg obs = decodeMsg(obs_msg);
            //RCLCPP_INFO(this->get_logger(), "Feature: type='%s', x=%.3f, y=%.3f, theta=%.3f, confidence=%.3f", 
            //obs.type.c_str(), obs.x, obs.y, obs.theta, obs.confidence);

            double sigma_x = std::sqrt(obs.covariance_pos[0][0]);
            double sigma_y = std::sqrt(obs.covariance_pos[1][1]);
            double sigma_theta = std::sqrt(obs.angle_variance);
            double sigma_pos = std::sqrt((sigma_x * sigma_x + sigma_y * sigma_y));

            std::normal_distribution<double> noise_pos(0.0, sigma_pos);
            std::normal_distribution<double> noise_theta(0.0, sigma_theta);

            double noise = noise_pos(generator_);
            double noisy_x = obs.x /* + noise */;
            double noisy_y = obs.y /* + noise; */;

            double measured_theta = obs.theta /* + noise_theta(generator_) */;

            with_angle_ = obs.with_angle;

            // ✅ ENHANCED: Add detailed debugging for angle measurements
            RCLCPP_DEBUG(this->get_logger(), "FEATURE_DEBUG: Processing feature - type='%s', detected_at=(%.3f,%.3f), measured_angle=%.3f rad (%.1f°), with_angle=%s, confidence=%.3f", 
                        obs.type.c_str(), noisy_x, noisy_y, measured_theta, measured_theta * 180.0 / M_PI, 
                        with_angle_ ? "YES" : "NO", obs.confidence);

            // Actually compute the feature likelihood
            double feature_likelihood = computeLikelihoodFeature(p, delta_scan_x, delta_scan_y, delta_scan_theta, noisy_x, noisy_y, measured_theta, sigma_pos, sigma_theta, obs.type);

            // DEBUG: Log feature likelihood computation
            RCLCPP_DEBUG(this->get_logger(), "PARTICLE_DEBUG: Particle %zu - feature_likelihood=%.6f, confidence=%.3f", 
                        particle_idx, feature_likelihood, obs.confidence);

            // Apply the reliability factor correctly
            likelihood += feature_likelihood * obs.confidence ;
        }

        double theta_degrees = p.theta * 180.0 / M_PI;

        // DEBUG: Log particle weight update
        double old_weight = p.weight;
        RCLCPP_DEBUG(this->get_logger(), "PARTICLE_DEBUG: Particle %zu at (%.3f, %.3f, %.1f°) - total_likelihood=%.6f, old_weight=%.6f", 
                    particle_idx, p.x, p.y, theta_degrees, likelihood, old_weight);

        // Gentle weight update instead of direct multiplication
        double learning_rate = 0.7; // How much to trust this measurement
        double new_weight = p.weight * likelihood;
        p.weight = learning_rate * new_weight + (1.0 - learning_rate) * p.weight;

        // DEBUG: Log weight after update
        RCLCPP_DEBUG(this->get_logger(), "PARTICLE_DEBUG: Particle %zu - new_weight=%.6f, final_weight=%.6f", 
                    particle_idx, new_weight, p.weight);

        // Less harsh space penalty
        bool penalize = !isParticleInFreeSpace(p.x, p.y, pgm, resolution, origin);
        if (penalize)
        {
            penalize*=0.01;
        }
        else
        {
            all_outside = false;
        }
    }

    if (all_outside == true)
    {
        RCLCPP_WARN(this->get_logger(), "⚠️ All particles are outside the free space - injecting random particles");
        injectRandomParticles_pgm(1); // Smaller injection
        return;
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "✅ Normalizing weights - particles are in valid space");
        normalizeWeights();
        
        // Print weight statistics after normalization
        double min_weight = std::numeric_limits<double>::max();
        double max_weight = 0.0;
        double avg_weight = 0.0;
        
        for (const auto &p : particles_) {
            min_weight = std::min(min_weight, p.weight);
            max_weight = std::max(max_weight, p.weight);
            avg_weight += p.weight;
        }
        avg_weight /= particles_.size();
        
        RCLCPP_INFO(this->get_logger(), "📈 Weight Stats: min=%.6f, max=%.6f, avg=%.6f, ratio=%.2f", 
                    min_weight, max_weight, avg_weight, max_weight/min_weight);
    }

    // Calculate ESS for resampling decision
    double ess = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0,
                                       [](double sum, const Particle &p)
                                       { return sum + (p.weight * p.weight); });
    RCLCPP_INFO(this->get_logger(), "📊 ESS Calculation: ESS=%.2f, num_particles=%d, ratio=%.3f", 
                ess, static_cast<int>(num_particles_), ess/num_particles_);

    // Adaptive ESS threshold based on number of features and their quality
    double base_threshold = resample_ess_threshold_;
    double adaptive_threshold = base_threshold;

    RCLCPP_INFO(this->get_logger(), "🎯 Thresholds: base=%.3f, adaptive=%.3f, features=%zu", 
                base_threshold, adaptive_threshold, msg->features.size());

    if (msg->features.size() < 2) // Few features
    {
        // adaptive_threshold *= 0.5; // Much more conservative
        RCLCPP_INFO(this->get_logger(), "⚠️ Few features detected (%zu), keeping conservative threshold", 
                    msg->features.size());
        
    }
    else{
    // **AGGRESSIVE: Use OR instead of AND, lower weight ratio threshold**
    bool ess_trigger = (ess <= num_particles_ * adaptive_threshold);
    double ess_threshold_value = num_particles_ * adaptive_threshold;

    RCLCPP_INFO(this->get_logger(), "🔄 Resampling Decision: ESS(%.2f) <= threshold(%.2f)? %s", 
                ess, ess_threshold_value, ess_trigger ? "YES - RESAMPLING" : "NO");

    if (ess_trigger)
    {
        RCLCPP_INFO(this->get_logger(), "🔄 TRIGGERING RESAMPLING: ESS too low");
        resampleParticles(ResamplingAmount::MAX_WEIGHT, ResamplingMethod::RESIDUAL);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "✅ NO RESAMPLING: ESS=%.1f is sufficient (threshold=%.1f)",
                    ess, ess_threshold_value);
    }}

    // Continuous small replacement to prevent stagnation
    // Always replace a small percentage of worst particles
    replaceWorstParticles_pgm(replace_worst_percentage_); // Replace 2% continuously
    
    // Compute updated pose after measurement update (but don't publish yet - timer will handle publishing)
    computeEstimatedPose();
    
    // REMOVED: Resample flag logic that was interfering with pose estimation

}

void ParticleFilter::resampleParticles(ResamplingAmount type, ResamplingMethod method)
{
    if (particles_.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No particles to resample.");
        return;
    }

    double max_weight = maxWeight();
    double ess = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0,
                                       [](double sum, const Particle &p)
                                       { return sum + (p.weight * p.weight); });

    RCLCPP_INFO(this->get_logger(), "RESAMPLING: Max weight: %.6f, ESS: %.1f", max_weight, ess);

    // REMOVED: resample_flag_ = true; - this was interfering with pose estimation

    // Perform resampling based on the specified method
    switch (method)
    {
    case ResamplingMethod::MULTINOMIAL:
        multinomialResample();
        break;
    case ResamplingMethod::STRATIFIED:
        stratifiedResample();
        break;
    case ResamplingMethod::SYSTEMATIC:
        systematicResample();
        break;
    case ResamplingMethod::RESIDUAL:
        residualResample();
        break;
    }

    // **IMPROVEMENT 4: Gentle weight reset with small noise for diversity**
/*     double base_weight = 1.0 / num_particles_;
    std::uniform_real_distribution<double> noise_dist(0.95, 1.05); // ±5% noise

    for (auto &p : particles_)
    {
        p.weight = base_weight * noise_dist(generator_); // Add small diversity
    } */

    // Normalize to ensure weights sum to 1
    normalizeWeights();

    // Remove cooldown - always allow particle-based estimation
    // resample_cooldown_counter_ = 2;  // REMOVED - this was causing pose to drift outside map
    
    RCLCPP_INFO(this->get_logger(), "🔄 Resampling completed - particles ready for pose estimation");

    // **IMPROVEMENT 5: More intelligent injection strategy**
    iterationCounter++;

    // Inject based on how desperate we are (lower ESS = more injection)
    bool should_inject = (iterationCounter >= inject_num_iterations_) || (ess < num_particles_ * 0.1);

    if (should_inject)
    {
        // Scale injection based on ESS - worse ESS = more injection
        double ess_ratio = ess / num_particles_;
        double injection_scale = std::max(0.5, 1.0 - ess_ratio * 2.0); // More injection when ESS is low
        double scaled_injection = inject_percentage_ * injection_scale;

        RCLCPP_INFO(this->get_logger(), "Injecting %.1f%% random particles (ESS-scaled from %.1f%%)",
                    scaled_injection * 100.0, inject_percentage_ * 100.0);

        injectRandomParticles_pgm(scaled_injection);
        iterationCounter = 0;
    }
}

void ParticleFilter::computeEstimatedPose()
{
    if (particles_.empty())
        return;

    // Calculate ESS to decide between odometry vs particle-based pose
    double ess = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0,
                                       [](double sum, const Particle &p)
                                       { return sum + (p.weight * p.weight); });
    
    double ess_ratio = ess / num_particles_;
    double ess_threshold = 0.8; // ESS threshold for switching between odometry and particle estimation
    
    if (ess_ratio > ess_threshold)
    {
        // 🚗 HIGH ESS: Good particle distribution - use odometry-based pose tracking
        // The pose was already updated with motion in motionUpdate(), so we just keep it
        pose_covariance_[0] = 0.05; // Lower covariance when tracking with odometry
        pose_covariance_[1] = 0.05;
        pose_covariance_[2] = 0.02;
        
        RCLCPP_DEBUG(this->get_logger(), "🚗 ODOMETRY-BASED POSE: ESS ratio %.3f > %.3f - using odometry tracking: (%.3f, %.3f, %.1f°)", 
                     ess_ratio, ess_threshold, x_last_final, y_last_final, theta_last_final * 180.0 / M_PI);
    }
    else
    {
        // 🎯 LOW ESS: Poor particle distribution - compute pose from particles
        std::vector<Particle> sorted_particles = particles_;
        std::sort(sorted_particles.begin(), sorted_particles.end(),
                  [](const Particle &a, const Particle &b)
                  {
                      return a.weight > b.weight;
                  });

        // Use only the top estimate_num_particles_ particles
        int num_top_particles = std::min(estimate_num_particles_, static_cast<int>(sorted_particles.size()));

        double x_sum = 0, y_sum = 0, theta_x_sum = 0, theta_y_sum = 0, weight_sum = 0;

        // First pass: calculate weight sum of top particles
        for (int i = 0; i < num_top_particles; i++)
        {
            weight_sum += sorted_particles[i].weight;
        }

        // Second pass: calculate weighted averages using only top particles
        for (int i = 0; i < num_top_particles; i++)
        {
            const auto &p = sorted_particles[i];
            double normalized_weight = p.weight / weight_sum;  // Renormalize to top particles only
            
            x_sum += p.x * normalized_weight;
            y_sum += p.y * normalized_weight;
            theta_x_sum += std::cos(p.theta) * normalized_weight;
            theta_y_sum += std::sin(p.theta) * normalized_weight;
        }

        // Update the global estimated pose from particles
        x_last_final = x_sum;
        y_last_final = y_sum;
        theta_last_final = std::atan2(theta_y_sum, theta_x_sum);

        if (theta_last_final > M_PI)
            theta_last_final -= 2 * M_PI;
        if (theta_last_final < -M_PI)
            theta_last_final += 2 * M_PI;

        // Higher covariance when estimating from particles
        pose_covariance_[0] = 0.15; // Higher covariance when using particles
        pose_covariance_[1] = 0.15;
        pose_covariance_[2] = 0.08;
        
        RCLCPP_DEBUG(this->get_logger(), "🎯 PARTICLE-BASED POSE: ESS ratio %.3f ≤ %.3f - computed from %d top particles: (%.3f, %.3f, %.1f°)", 
                     ess_ratio, ess_threshold, num_top_particles, x_last_final, y_last_final, theta_last_final * 180.0 / M_PI);
    }
}

// publish the estimated pose and the map to odom transform
void ParticleFilter::publishEstimatedPose()
{
    if (particles_.empty())
        return;

    // Simply publish the current estimated pose (already computed in motion/measurement updates)
    RCLCPP_DEBUG(this->get_logger(), "📍 PUBLISHING POSE: (%.3f, %.3f, %.1f°)", 
                 x_last_final, y_last_final, theta_last_final * 180.0 / M_PI);

    if (!msg_odom_base_link_)
    {
        RCLCPP_WARN(this->get_logger(), "Skipping pose publication: No odometry data available.");
        return;
    }

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = x_last_final;
    pose_msg.pose.pose.position.y = y_last_final;

    tf2::Quaternion q;
    q.setRPY(0, 0, theta_last_final);
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    pose_msg.pose.covariance[0] = pose_covariance_[0]; // Covariance of x
    pose_msg.pose.covariance[7] = pose_covariance_[1]; // Covariance of y
    pose_msg.pose.covariance[35] = pose_covariance_[2]; // Covariance of theta
    pose_pub_->publish(pose_msg);

    // Publish `map -> base_link` transform
    geometry_msgs::msg::TransformStamped map_to_pose_tf;
    map_to_pose_tf.header.stamp = this->get_clock()->now();
    map_to_pose_tf.header.frame_id = "map";
    map_to_pose_tf.child_frame_id = "estimated_pose";

    // Calculate map -> odom transform
    map_to_pose_tf.transform.translation.x = x_last_final;
    map_to_pose_tf.transform.translation.y = y_last_final;
    map_to_pose_tf.transform.translation.z = 0.0;

    map_to_pose_tf.transform.rotation.x = q.x();
    map_to_pose_tf.transform.rotation.y = q.y();
    map_to_pose_tf.transform.rotation.z = q.z();
    map_to_pose_tf.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(map_to_pose_tf);

    // RCLCPP_INFO(this->get_logger(), "Published estimated pose (Top 10 weighted particles).");
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilter>());
    rclcpp::shutdown();
    return 0;
}

#pragma endregion pf functions

//! Particle Filter Functions !//