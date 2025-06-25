#include "robot_localization_package/particle_filter.hpp"
#include "tf2/utils.h"

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
                                    msg_odom_base_link_(nullptr), last_map_msg_(nullptr)
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
     
    // Initialize some variables
    init_weight = 1.0 / num_particles_;

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

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&ParticleFilter::motionUpdate, this, std::placeholders::_1));

    // Subscribe to the features observed by the robot and odometry topics
    feature_sub_ = this->create_subscription<robot_msgs::msg::FeatureArray>(
        "/features", 10,
        std::bind(&ParticleFilter::storeMapMessage, this, std::placeholders::_1));

    // Create publishers for the estimated pose and particles
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/estimated_pose", 10);
    particles_color_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/particles_color", 10);
    particles_no_color_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/particles_no_color", 10);

    // Create a transform broadcaster
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Create a timer to publish the estimated pose and particles
    timer_pose_ = create_wall_timer(std::chrono::milliseconds(500), std::bind(&ParticleFilter::publishEstimatedPose, this));

    // Initialize the color pallete for the particles weights
    computeColorWeightLookup();

    // Initialize the particles
    initializeParticles_pgm();

    while (rclcpp::ok() && !last_map_msg_)
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for the first keypoint message...");
        rclcpp::spin_some(this->get_node_base_interface());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

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

// store the map message received from the topic
void ParticleFilter::storeMapMessage(const robot_msgs::msg::FeatureArray::SharedPtr msg)
{
    // lets save the timestamp
    /* last_map_msg_ = msg;
    stored_features_are_valid = true;
    RCLCPP_INFO(this->get_logger(), "Received features"); */
    
    // Received Features
    last_map_msg_ = msg;

    double delta_x, delta_y, delta_theta, delta_distance;

    delta_x = odom_x - last_update_x_;
    delta_y = odom_y - last_update_y_;
    delta_theta = abs(odom_theta - last_update_theta_);
    delta_distance = std::hypot(delta_x, delta_y);
    RCLCPP_INFO(this->get_logger(), "delta_x: %.2f", delta_x);

    if (delta_distance > motion_delta_distance_ || delta_theta > motion_delta_angle_)
    {
        // Update the particle weights and reset the robot motion logic
        RCLCPP_INFO(this->get_logger(), "MeasureUpdate!");
        measurementUpdate(last_map_msg_);
        last_update_x_ = odom_x;
        last_update_y_ = odom_y;
        last_update_theta_ = odom_theta; 
    }
}

std::vector<map_features::Feature> ParticleFilter::getExpectedFeatures(const Particle &p, const std::string &type)
{
    std::vector<map_features::Feature> features_particle;

    double cos_theta = std::cos(p.theta);
    double sin_theta = std::sin(p.theta);

    for (const auto &feature_ptr : global_features_)
    {
        if (feature_ptr->type == type)
        {
            auto object_ptr = std::dynamic_pointer_cast<map_features::Feature>(feature_ptr);
            if (!object_ptr)
                continue;

            double map_x = object_ptr->x;
            double map_y = object_ptr->y;
            double feature_theta = object_ptr->theta;

            double particle_x = cos_theta * (map_x - p.x) + sin_theta * (map_y - p.y);
            double particle_y = -sin_theta * (map_x - p.x) + cos_theta * (map_y - p.y);

            features_particle.emplace_back(particle_x, particle_y, feature_theta, type);
        }
    }

    return features_particle;
}

// transform angle from the map frame to the particle frame
double ParticleFilter::transformAngleToParticleFrame(double feature_theta_map, double particle_theta)
{

    // print the theta of the feature and the particle
    // RCLCPP_INFO(this->get_logger(), "Feature theta: %.2f, Particle theta: %.2f", feature_theta_map, particle_theta);

    if (particle_theta < -M_PI)
        particle_theta += 2 * M_PI;
    else if (particle_theta > M_PI)
        particle_theta -= 2 * M_PI;

    feature_theta_map = feature_theta_map * (M_PI / 180.0);
    if (feature_theta_map < -M_PI)
        feature_theta_map += 2 * M_PI;
    else if (feature_theta_map > M_PI)
        feature_theta_map -= 2 * M_PI;

    // compute the angle between the feature and the particle

    double angle = feature_theta_map - particle_theta;

    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle < -M_PI)
        angle += 2 * M_PI;

    return angle;
}

// compute the likelihood for the orientation of the corner feature
double ParticleFilter::computeAngleLikelihood(double measured_angle, double expected_angle, double sigma)
{
    if (measured_angle > M_PI)
        measured_angle -= 2 * M_PI;
    if (measured_angle < -M_PI)
        measured_angle += 2 * M_PI;

    double error = measured_angle - expected_angle;

    while (error > M_PI)
        error -= 2 * M_PI;
    while (error < -M_PI)
        error += 2 * M_PI;

    // double coeff = 1.0 / std::sqrt(2.0 * M_PI * sigma * sigma);
    double exponent = -0.5 * (error * error) / (10 * sigma * sigma);

    return std::exp(exponent);
}

// compute the likelihood of a corner feature based on distance and angle
double ParticleFilter::computeLikelihoodFeature(const Particle &p, double noisy_x, double noisy_y, double measured_theta, double sigma_pos, double sigma_theta, const std::string &type)
{
    std::vector<map_features::Feature> expected_features = getExpectedFeatures(p, type);

    double min_dist = std::numeric_limits<double>::max();
    map_features::Feature best_feature(0, 0, 0, type);

    double likelihood = 0.0;

    for (const auto &exp : expected_features)
    {
        double dist = std::hypot(noisy_x - exp.x, noisy_y - exp.y);
        if (dist < min_dist)
        {
            min_dist = dist;
            best_feature = exp;
        }
    }

    // Compute likelihood based on distance and angle
    double expected_feature_angle = transformAngleToParticleFrame(best_feature.theta, p.theta);
    double angle_likelihood = computeAngleLikelihood(measured_theta, expected_feature_angle, sigma_theta);
    double distance_likelihood = std::exp(-(min_dist * min_dist) / (sigma_pos * sigma_pos));

    if (with_angle_)
    {
        likelihood = (angle_likelihood * distance_likelihood);
    }
    else
    {
        likelihood = distance_likelihood;
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
    //RCLCPP_INFO(this->get_logger(), "Motion!");
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

    // update particles if significant motion is detected
    if (delta_distance > motion_delta_distance_ || std::abs(delta_theta_odom) > motion_delta_angle_)
    {
        //RCLCPP_INFO(this->get_logger(), "Motion Update!");
        for (auto &p : particles_)
        {
            p.x += delta_x_robot * std::cos(p.theta) - delta_y_robot * std::sin(p.theta) + noise_x(generator_);
            p.y += delta_x_robot * std::sin(p.theta) + delta_y_robot * std::cos(p.theta) + noise_y(generator_);
            p.theta += delta_theta_odom + noise_theta(generator_);

            if (p.theta > M_PI)
                p.theta -= 2 * M_PI;
            if (p.theta < -M_PI)
                p.theta += 2 * M_PI;

            bool penalize = !isParticleInFreeSpace(p.x, p.y, pgm, resolution, origin);
            if (penalize)
            {
                //initializeParticle(p, p.weight);
                p.weight = p.weight / 4;
            }
        }
        
        //RCLCPP_INFO(this->get_logger(), "Motion Update after particles!");

        last_odom_x_ = odom_x;
        last_odom_y_ = odom_y;
        last_odom_theta_ = odom_theta;

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

    rclcpp::Time scan_timestamp = msg->header.stamp;

    geometry_msgs::msg::TransformStamped tf;

    try {
        tf = tf_buffer_->lookupTransform(
            "odom",              // target frame
            "base_footprint",    // source frame
            scan_timestamp,                   // timestamp
            rclcpp::Duration::from_seconds(0.1)  // timeout
        );

        // Convert to PoseStamped
        double scan_pose_x, scan_pose_y, scan_pose_theta;

        scan_pose_x = tf.transform.translation.x;
        scan_pose_y = tf.transform.translation.y;
        scan_pose_theta = tf2::getYaw(tf.transform.rotation);


        // Now pose contains the pose of base_footprint in odom frame at time x
        } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
    }


    for(auto &p : particles_)
    {
        double likelihood = 0;

        for (const auto &obs_msg : msg->features)
        {
            DecodedMsg obs = decodeMsg(obs_msg);

            double sigma_x = std::sqrt(obs.covariance_pos[0][0]);
            double sigma_y = std::sqrt(obs.covariance_pos[1][1]);
            double sigma_theta = std::sqrt(obs.angle_variance);
            double sigma_pos = std::sqrt((sigma_x * sigma_x + sigma_y * sigma_y));

            // Compute likelihood based on feature type

            likelihood += computeLikelihoodFeature(p, obs.x, obs.y, obs.theta, sigma_pos, sigma_theta, obs.type);
        }

        p.weight *= likelihood;

    }

    RCLCPP_INFO(this->get_logger(), "Measurement before normalize!");
    normalizeWeights();

    // perform resampling
    //RCLCPP_INFO(this->get_logger(), "Measurement before resample!");
    resampleParticles(ResamplingAmount::ESS, ResamplingMethod::RESIDUAL);
    //RCLCPP_INFO(this->get_logger(), "Measurement after resample!");


    // Replace worst particles if resampling flag is not set
    if (!resample_flag_)
    {
        // replaceWorstParticles(replace_worst_percentage_);
        replaceWorstParticles_pgm(replace_worst_percentage_);
    }
    else
    {
        resample_flag_ = false;
    }
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

    RCLCPP_INFO(this->get_logger(), "Max weight: %f, ESS: %f", max_weight, ess);

    switch (type)
    {
    case ResamplingAmount::ESS:
        if (ess > num_particles_ * resample_ess_threshold_)
        {
            RCLCPP_INFO(this->get_logger(), "Skipping resampling, particles are well-distributed.");
            return;
        }
        break;
    case ResamplingAmount::MAX_WEIGHT:
        if (max_weight < resample_max_weight_threshold_ / num_particles_)
        {
            RCLCPP_INFO(this->get_logger(), "Skipping resampling, max weight is not high enough.");
            return;
        }
        break;
    }

    resample_flag_ = true;

    // Perform resampling based on the specified method
    switch (method)
    {
    case ResamplingMethod::RESIDUAL:
        residualResample();
        break;
    }

    // Reset particle weights after resampling
    for (auto &p : particles_)
    {
        p.weight = init_weight;
    }

    // Inject random particles base on number of resamples performed
    iterationCounter++;
    if (iterationCounter == inject_num_iterations_)
    {
        RCLCPP_INFO(this->get_logger(), "Injecting random particles.");
        // injectRandomParticles(inject_percentage_);
        injectRandomParticles_pgm(inject_percentage_);

        iterationCounter = 0;
    }
}

// compute the estimated pose based on the top-weighted particles
void ParticleFilter::computeEstimatedPose()
{
    if (particles_.empty())
        return;

    std::vector<Particle> sorted_particles = particles_;
    std::sort(sorted_particles.begin(), sorted_particles.end(),
              [](const Particle &a, const Particle &b)
              {
                  return a.weight > b.weight;
              });

    // Use only the top estimate_num_particles_ particles
    int num_top_particles = std::min(estimate_num_particles_, static_cast<int>(sorted_particles.size()));

    double x_sum = 0, y_sum = 0, theta_x_sum = 0, theta_y_sum = 0, weight_sum = 0;

    for (int i = 0; i < num_top_particles; i++)
    {
        const auto &p = sorted_particles[i];
        x_sum += p.x * p.weight;
        y_sum += p.y * p.weight;
        theta_x_sum += std::cos(p.theta) * p.weight;
        theta_y_sum += std::sin(p.theta) * p.weight;
        weight_sum += p.weight;
    }

    if (weight_sum > 0)
    {
        x_sum /= weight_sum;
        y_sum /= weight_sum;
        theta_x_sum /= weight_sum;
        theta_y_sum /= weight_sum;
    }

    // Compute the final estimated pose
    x_last_final = x_sum;
    y_last_final = y_sum;
    theta_last_final = std::atan2(theta_y_sum, theta_x_sum);

    if (theta_last_final > M_PI)
        theta_last_final -= 2 * M_PI;
    if (theta_last_final < -M_PI)
        theta_last_final += 2 * M_PI;

    // --- Covariance computation ---
    double cov_xx = 0, cov_yy = 0, cov_tt = 0;
    double cov_xy = 0, cov_xt = 0, cov_yt = 0;

    for (int i = 0; i < num_top_particles; i++)
    {
        const auto &p = sorted_particles[i];
        double dx = p.x - x_last_final;
        double dy = p.y - y_last_final;

        // Wrap angle difference to [-pi, pi]
        double dtheta = p.theta - theta_last_final;
        while (dtheta > M_PI) dtheta -= 2 * M_PI;
        while (dtheta < -M_PI) dtheta += 2 * M_PI;

        cov_xx += p.weight * dx * dx;
        cov_yy += p.weight * dy * dy;
        cov_tt += p.weight * dtheta * dtheta;

    }

    if (weight_sum > 0)
    {
        cov_xx /= weight_sum;
        cov_yy /= weight_sum;
        cov_tt /= weight_sum;
    }

    pose_covariance_[0] = cov_xx; // Covariance of x
    pose_covariance_[1] = cov_yy; // Covariance of y
    pose_covariance_[2] = cov_tt; // Covariance of theta
}

// publish the estimated pose and the map to odom transform
void ParticleFilter::publishEstimatedPose()
{
    if (particles_.empty())
        return;

    computeEstimatedPose();

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