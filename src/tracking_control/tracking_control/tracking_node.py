import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
class SimpleKalmanFilter:
    def __init__(self, dim_x, dim_z, process_noise, measurement_noise):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))  # State vector
        self.F = np.eye(dim_x)         # State transition matrix
        self.H = np.zeros((dim_z, dim_x))  # Measurement function
        self.P = np.eye(dim_x) * 1000  # Covariance matrix
        self.Q = np.eye(dim_x) * process_noise  # Process noise covariance matrix
        self.R = np.eye(dim_z) * measurement_noise  # Measurement noise covariance matrix
        self.K = np.zeros((dim_x, dim_z))  # Kalman gain

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(self.K, y)
        self.P = self.P - np.dot(np.dot(self.K, self.H), self.P)

def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obj_pose = None

        # Last known object pose
        self.last_known_obj_pose = None
        
        #Dyamic linear gain
        self.linear_gain_base = 1
        
        #Dynamic angular gain
        self.angular_gain_base = 1.1
        
        #stop_distance
        self.stop_distance = 0.45
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_obj_pose = self.create_subscription(PoseStamped, '/detected_color_object_pose', self.detected_obj_pose_callback, 10)
    
        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)

        self.kf = SimpleKalmanFilter(dim_x=4, dim_z=2, process_noise=1, measurement_noise=1)
        self.kf.x = np.array([0., 0., 0., 0.])
        self.kf.F = np.array([[ 1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *=1000.
        self.kf.R =np.array([[1, 0],
                            [0, 1]])
        self.kf.Q = np.eye(4)
    
    def detected_obj_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # Update the last know pose (non filtered)
        self.last_known_obj_pose = center_points
        
        max_distance = 2 # the maximimum (limit) distance in meters 
        max_height = 0.7 # the maximum (limit) height in meters
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        if np.linalg.norm(center_points) > max_distance or center_points[2] > max_height :
             return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obj_pose = cp_world

        z = np.array([msg.pose.position.x, msg.pose.position.y])
        self.kf.predict()
        self.kf.update(z)

        self.obj_pose = self.kf.x[:2].flatten()
        
        self.get_logger().info(f"Kalman Filter State: {self.kf.x.flatten()}")
        self.get_logger().info(f"Object Pose: {self.obj_pose}")
        
    def get_current_object_pose(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            if len(self.obj_pose) == 2:
                self.obj_pose = np.append(self.obj_pose, 0)  # Append a zero if it's a 2-element vector
            elif len(self.obj_pose) > 3:
                self.obj_pose = self.obj_pose[:3]  # Truncate to 3 elements if it's longer

            object_pose = robot_world_R @ self.obj_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
            
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return 
        
        return object_pose
    
    def timer_update(self):
        ################### Write your code here ###################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly
        if self.obj_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            #Check if there is last know pose
            if self.last_known_obj_pose is not None:
                
                # Determine the direction to turn based on last known position
                angle_to_last_known_pos = math.atan2(self.last_known_obj_pose[1], self.last_known_obj_pose[0])
                cmd_vel.angular.z = 0.4 * angle_to_last_known_pos / abs(angle_to_last_known_pos) 
            else:
                # If there is no known last positin, turn in default direction
                 cmd_vel.angular.z = max(0.0, cmd_vel.angular.z - 0.1)
                
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        # Get the current object pose in the robot base_footprint frame
        current_object_pose = self.get_current_object_pose()
        if current_object_pose is None:
            return
        
        # TODO: get the control velocity command
        cmd_vel = self.controller()
        
        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########
        
        # TODO: Update the control velocity command

        #Dynamic gain adjustment factor
        linear_gain_factor = 0.8
        angular_gain_factor = 0.7

       

        #Calculate distance and angle to the object
        distance = np.linalg.norm(self.obj_pose[:2]) #distance based on x and y mesurments
        angle = math.atan2(self.obj_pose[1], self.obj_pose[0]) #Angle to object

        linear_gain = self.linear_gain_base -linear_gain_factor * (distance - self.stop_distance)
        angular_gain = self.angular_gain_base + angular_gain_factor * abs(angle)

        linear_gain=max(linear_gain, 0.1)
        angular_gain = max(angular_gain, 0.1)
        
        cmd_vel = Twist()

        #Check if we are close enough to stop location
        if distance > self.stop_distance:
            # Adjust linear velocity based on distance, reduces speed as it gets closer
            cmd_vel.linear.x = linear_gain * ( distance - self.stop_distance )

            #Adjust angular velocity based on the angle, sharper turn for larger angles
            cmd_vel.angular.z = angular_gain * angle
        
        else:
            #Stop moving if close enough
            cmd_vel.linear.x = 0
            cmd_vel.angular.z =0

        
            
        self.get_logger().info(f"Distance: {distance}, Angle: {angle}")
        self.get_logger().info(f"Linear Gain: {linear_gain}, Angular Gain: {angular_gain}")
        self.get_logger().info(f"Command Velocity - Linear: {cmd_vel.linear.x}, Angular: {cmd_vel.angular.z}")
        self.get_logger().info(f"Angular Gain Factor: {angular_gain_factor}, Angle: {angle}, Angular Gain: {angular_gain}")
        
        return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
