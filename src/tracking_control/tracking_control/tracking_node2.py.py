import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.subscription = self.create_subscription(
            PoseStamped, 
            '/detected_color_object_pose', 
            self.object_pose_callback, 
            10)
        self.publisher = self.create_publisher(Path, '/navigation_path', 10)
        self.current_path = None

    def object_pose_callback(self, msg):
        self.get_logger().info(f'Object detected at {msg.pose.position}')
        # Pause path following
        self.pause_navigation()
        # Move to the object
        self.move_to_object(msg.pose)
        # Perform interaction
        self.perform_interaction()
        # Resume path following
        self.resume_navigation()

    def pause_navigation(self):
        # Logic to pause the robot's path navigation
        pass

    def move_to_object(self, pose):
        # Logic to move the robot towards the detected object
        pass

    def perform_interaction(self):
        # Logic to perform the desired interaction, like stopping for a few seconds
        rclpy.sleep(5)  # Example: stop for 5 seconds

    def resume_navigation(self):
        # Logic to resume the navigation along the previous path
        if self.current_path:
            self.publisher.publish(self.current_path)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()