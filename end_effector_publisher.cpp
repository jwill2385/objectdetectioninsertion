#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_interface/planning_request.h>
#include <moveit/planning_interface/planning_response.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <string.h>
#include <cmath>
#include <tuple>
using namespace std::chrono_literals;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");
static const std::string PLANNING_GROUP = "ur_manipulator";
/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
      
      publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("end_effector_talker", 10);
      timer_ = this->create_wall_timer(
      10ms, std::bind(&MinimalPublisher::timer_callback, this));
      // 2ms timer means callback executes 500 times a second
    }

  private:
    void timer_callback()
    {
      RCLCPP_INFO(LOGGER, "Call BACK");
      //Try to initialize Move it here
      rclcpp::NodeOptions node_options;
      node_options.automatically_declare_parameters_from_overrides(true);
      auto move_group_node = rclcpp::Node::make_shared("move_group_end_effect", node_options);
        // We spin up a SingleThreadedExecutor for the current state monitor to get information
      // about the robot's state.
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(move_group_node);
      std::thread([&executor]() { executor.spin(); }).detach();

      //static const std::string PLANNING_GROUP = "ur_manipulator";
      moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP);
      home_pose = move_group.getCurrentPose().pose;
      auto pose_message = geometry_msgs::msg::Pose();
      //home_pose = current_move_group.getCurrentPose().pose;
      pose_message = home_pose;
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", std::to_string(count_++));
      publisher_->publish(pose_message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    int num_;
    //moveit::planning_interface::MoveGroupInterface current_move_group;
    geometry_msgs::msg::Pose home_pose;
    size_t count_;
    //NOTE: Try to add new pose variable here
};

int main(int argc, char * argv[])
{
  // Step 1 connect to moveit and get pose data
  RCLCPP_INFO(LOGGER, "Hello lets start");
  rclcpp::init(argc, argv);
  // rclcpp::NodeOptions node_options;
  // node_options.automatically_declare_parameters_from_overrides(true);
  // auto move_group_node = rclcpp::Node::make_shared("move_group_end_effect", node_options);
  // RCLCPP_INFO(LOGGER, "NODE CREATED");

  // We spin up a SingleThreadedExecutor for the current state monitor to get information
  // about the robot's state.
  // rclcpp::executors::SingleThreadedExecutor executor;
  // executor.add_node(move_group_node);
  // std::thread([&executor]() { executor.spin(); }).detach();

  //static const std::string PLANNING_GROUP = "ur_manipulator";
  // moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP);
  // RCLCPP_INFO(LOGGER, "Executor CREATED");
  // moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP);

  // move_group.setGoalTolerance(0.05);
  //geometry_msgs::msg::Pose home_pose;

  // home_pose = move_group.getCurrentPose().pose;
  // RCLCPP_INFO(LOGGER, "Curent position");

  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.orientation.w));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.orientation.x));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.orientation.y));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.orientation.z));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.position.x));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.position.y));
  // RCLCPP_INFO(LOGGER, std::to_string(home_pose.position.z));

  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}