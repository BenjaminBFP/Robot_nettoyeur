__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"


import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

import numpy as np

from .my_common import *    #common variables are stored here


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        #initialize attributes
        self.agents_pose = [None]*self.nb_agents    #[(x_1, y_1), (x_2, y_2), (x_3, y_3)] if there are 3 agents
        self.x = self.y = self.yaw = self.n = self.angle_increment = self.angle_max = self.angle_min = self.ranges = self.theta = self.range_max = None   #the pose of this specific agent running the node
        self.avoid_target_yaw = None  # target yaw for pi/2 avoidance rotation
        self.avoid_phase = 'none'  # 'none' | 'rotating' | 'moving_clear'

        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.init_map()

        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own LIDAR topic
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.2s of period <=> 5 Hz
        self.create_timer(0.5, self.strategy)      #0.5s of period <=> 2 Hz
        self.create_timer(1, self.publish_maps) #1Hz
    

    def load_params(self):
        """ Load parameters from launch file """
        self.declare_parameters(    #A node has to declare ROS parameters before getting their values from launch files
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),    #robot's namespace: either 1, 2 or 3
                ("robot_size", rclpy.Parameter.Type.DOUBLE),    #robot's diameter in meter
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),   #environment dimensions (width height)
                ("nb_agents", rclpy.Parameter.Type.INTEGER),    #total number of agents (this agent included) to map the environment
            ]
        )

        #Get launch file parameters related to this node
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value
    

    def init_map(self):
        """ Initialize the map to share with others if it is bot_1 """
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"    #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_msg.header.stamp = self.get_clock().now().to_msg() #get the current ROS time to send the msg
        self.map_msg.info.resolution = self.robot_size  #Map cell size corresponds to robot size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)   #nb of rows
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)    #nb of columns
        self.map_msg.info.origin.position.x = -self.env_size[1]/2   #x and y coordinates of the origin in map reference frame
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0    #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE #all the cells are unexplored initially
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height  
    

    def merged_map_cb(self, msg):
        """ 
            Get the current common map and update ours accordingly.
            This method is automatically called whenever a new message is published on the topic /merged_map.
            'msg' is a nav_msgs/msg/OccupancyGrid message.
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                # if received_map[i, j] != UNEXPLORED_SPACE_VALUE:
                    self.map[i, j] = received_map[i, j]


    def odom1_cb(self, msg):
        """ 
            @brief Get agent 1 position.
            This method is automatically called whenever a new message is published on topic /bot_1/odom.
            
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)
        # self.get_logger().info(f"Agent 1: ({x:.2f}, {y:.2f})")
    

    def odom2_cb(self, msg):
        """ 
            @brief Get agent 2 position.
            This method is automatically called whenever a new message is published on topic /bot_2/odom.
             
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)
        # self.get_logger().info(f"Agent 2: ({x:.2f}, {y:.2f})")


    def odom3_cb(self, msg):
        """ 
            @brief Get agent 3 position.
            This method is automatically called whenever a new message is published on topic /bot_3/odom.
            
            @param msg This is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)
        # self.get_logger().info(f"Agent 3: ({x:.2f}, {y:.2f})")


    def map_update(self):
        """ Consider sensor readings to update the agent's map """

        if self.ranges is None or self.x is None:
           return
        
        xp_m = []
        yp_m = []
        angles = np.linspace(self.angle_min, self.angle_max, len(self.ranges), endpoint=False)
        detected_points = []    
        # self.get_logger().info(f"self.x: {self.x}, self.y: {self.y}")
        for i, r in enumerate(self.ranges) :
            if np.isinf(r) :
                detected_points.append(0)
                self.ranges[i] = self.range_max
                r = self.range_max
            else :
                detected_points.append(1)
            xp_m.append(r * np.cos(angles[i])*np.cos(self.yaw) - r * np.sin(angles[i])*np.sin(self.yaw)+ self.x)
            yp_m.append(-r * np.sin(angles[i])*np.cos(self.yaw) - r * np.cos(angles[i])*np.sin(self.yaw)- self.y)

        
        resolution = self.map_msg.info.resolution
        grid_size_x = self.w
        grid_size_y = self.h
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y
        robot_i = int((self.x - origin_x) / resolution)
        robot_j = int((-self.y - origin_y) / resolution)

        for r, x, y, detected in zip(self.ranges, xp_m, yp_m, detected_points):                

                # conversion
                i = int(((x - origin_x) / resolution))
                j = int(((y - origin_y) / resolution))

                # hors map
                if not (0 <= i < grid_size_x and 0 <= j < grid_size_y):
                    continue

                # obstacle
                if detected == 1:
                    self.map[j, i] = OBSTACLE_VALUE

                # espace libre 
                num = max(abs(i - robot_i), abs(j - robot_j))

                if num == 0:
                    continue

                for k in range(num):

                    xi = int(robot_i + (i - robot_i) * k / num)
                    yj = int(robot_j + (j - robot_j) * k / num)

                    if 0 <= xi < grid_size_x and 0 <= yj < grid_size_y:

                        # ne pas Ã©craser un obstacle
                        if self.map[yj, xi] != OBSTACLE_VALUE:
                            self.map[yj, xi] = FREE_SPACE_VALUE


    def lidar_cb(self, msg):
        """ 
            @brief Get messages from LIDAR topic.
            This method is automatically called whenever a new message is published on topic /bot_x/laser/scan, where 'x' is either 1, 2 or 3.
            
            @param msg This is a sensor_msgs/msg/LaserScan message.
        """
        self.ranges = msg.ranges
        self.n = len(self.ranges)
        self.angle_increment = msg.angle_increment
        self.angle_max = msg.angle_max
        self.angle_min = msg.angle_min
        self.range_max = msg.range_max
        
        pass

    def publish_maps(self):
        """ 
            Publish updated map to topic /bot_x/map, where x is either 1, 2 or 3.
            This method is called periodically (1Hz) by a ROS2 timer, as defined in the constructor of the class.
        """
        self.map_msg.data = np.flipud(self.map).flatten().tolist()  #transform the 2D array into a list to publish it
        self.map_agent_pub.publish(self.map_msg)    #publish map to other agents


    def strategy(self):
        """
        Decision and action layers.
        Frontier-based exploration (slides 54-69) with LIDAR wall avoidance.

        Wall avoidance logic:
          - Only the FRONT sector (±45°) triggers avoidance.
          - The robot turns 90° in the trigonometric direction (CCW) relative
            to the angle of the closest front beam.
          - Side obstacles do NOT stop the robot; navigation naturally steers
            away from them via heading control toward the frontier.
        """
        cmd = Twist()

        if self.ranges is None or self.x is None or self.yaw is None:
            return

        # ------------------------------------------------------------------ #
        #  PARAMETERS                                                          #
        # ------------------------------------------------------------------ #
        WALL_THRESHOLD = 0.6    # [m] — only react when truly close in front
        LINEAR_SPEED   = 0.3    # [m/s]
        ANGULAR_SPEED  = 0.5    # [rad/s]
        GOAL_THRESHOLD = 0.5    # [m] consider frontier reached
        NAV_KP         = 1.2    # heading proportional gain


        # ------------------------------------------------------------------ #
        #  1. WALL AVOIDANCE                                                  #
        #                                                                     #
        #  Si on tourne : on tourne, point. Pas de LIDAR, pas d'avance.      #
        #  Une fois pi/2 atteint : on regarde devant et on avance.           #
        # ------------------------------------------------------------------ #

        # --- En cours de rotation : on tourne jusqu'au bout, rien d'autre ---
        if self.avoid_phase == 'rotating':
            yaw_error = np.arctan2(
                np.sin(self.avoid_target_yaw - self.yaw),
                np.cos(self.avoid_target_yaw - self.yaw)
            )
            if abs(yaw_error) > 0.05:
                cmd.linear.x  = 0.0
                cmd.angular.z = float(ANGULAR_SPEED * np.sign(yaw_error))
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info(f"[AVOID] rotating yaw_err={np.degrees(yaw_error):.1f} deg")
                return
            else:
                self.get_logger().info("[AVOID] pi/2 done — resuming.")
                self.avoid_phase = 'none'

        # --- Phase normale : on regarde devant et on décide ---
        ranges_arr = np.array(self.ranges, dtype=float)
        angles     = np.linspace(self.angle_min, self.angle_max,
                                 len(ranges_arr), endpoint=False)
        valid      = np.isfinite(ranges_arr)

        front_mask   = np.abs(angles) <= np.pi / 12  # ±15°
        front_valid  = front_mask & valid
        front_ranges = np.where(front_valid, ranges_arr, np.inf)
        min_front    = float(np.min(front_ranges)) if front_valid.any() else np.inf

        if min_front <= WALL_THRESHOLD:
            # Déclencher une rotation exacte de pi/2 CCW (trigonométrique)
            self.avoid_target_yaw = np.arctan2(
                np.sin(self.yaw + np.pi / 2.0),
                np.cos(self.yaw + np.pi / 2.0)
            )
            self.avoid_phase  = 'rotating'
            cmd.linear.x  = 0.0
            cmd.angular.z = float(ANGULAR_SPEED)
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(
                f"[AVOID] wall at {min_front:.2f}m — pi/2 rotation started "
                f"target_yaw={np.degrees(self.avoid_target_yaw):.1f} deg"
            )
            return

        # ------------------------------------------------------------------ #
        #  2. FRONTIER DETECTION                                              #
        # ------------------------------------------------------------------ #
        m          = self.map
        unexplored = (m == UNEXPLORED_SPACE_VALUE)
        free       = (m == FREE_SPACE_VALUE)

        has_free_nb = (
            np.roll(free,  1, axis=0) | np.roll(free, -1, axis=0) |
            np.roll(free,  1, axis=1) | np.roll(free, -1, axis=1)
        )
        has_free_nb[ 0, :] = False
        has_free_nb[-1, :] = False
        has_free_nb[:,  0] = False
        has_free_nb[:, -1] = False

        frontier_rows, frontier_cols = np.where(unexplored & has_free_nb)

        if frontier_rows.size == 0:
            self.get_logger().info("[DONE] No frontiers left — exploration complete.")
            self.cmd_vel_pub.publish(cmd)
            return

        # ------------------------------------------------------------------ #
        #  3. NEAREST FRONTIER                                                #
        # ------------------------------------------------------------------ #
        resolution = self.map_msg.info.resolution
        origin_x   = self.map_msg.info.origin.position.x
        origin_y   = self.map_msg.info.origin.position.y

        wx_all = frontier_cols * resolution + origin_x
        wy_all = frontier_rows * resolution + origin_y  # j = (y - origin_y)/res  →  y = row*res + origin_y

        dists  = np.hypot(wx_all - self.x, wy_all - self.y)
        best   = int(np.argmin(dists))
        tx, ty = float(wx_all[best]), float(wy_all[best])

        # ------------------------------------------------------------------ #
        #  4. NAVIGATE TO FRONTIER                                            #
        # ------------------------------------------------------------------ #
        dx   = tx - self.x
        dy   = ty - self.y
        dist = np.hypot(dx, dy)

        if dist < GOAL_THRESHOLD:
            self.cmd_vel_pub.publish(cmd)
            return

        target_angle  = np.arctan2(dy, dx)
        heading_error = np.arctan2(
            np.sin(target_angle - self.yaw),
            np.cos(target_angle - self.yaw)
        )

        # Toujours avancer (même vitesse réduite si grosse erreur d'angle)
        # pour éviter de tourner sur place indéfiniment
        if abs(heading_error) > np.pi / 2:
            # Erreur > 90° : rotation pure, cible vraiment derrière
            cmd.linear.x  = 0.0
            cmd.angular.z = float(ANGULAR_SPEED * np.sign(heading_error))
        else:
            # Avance en corrigeant le cap, vitesse proportionnelle à l'alignement
            scale = 1.0 - abs(heading_error) / (np.pi / 2)
            cmd.linear.x  = float(LINEAR_SPEED * scale)
            cmd.angular.z = float(NAV_KP * heading_error)

        self.get_logger().info(
            f"[NAV] pos=({self.x:.2f},{self.y:.2f}) yaw={np.degrees(self.yaw):.1f}° "
            f"target=({tx:.2f},{ty:.2f}) target_angle={np.degrees(target_angle):.1f}° "
            f"err={np.degrees(heading_error):.1f}° "
            f"lin={cmd.linear.x:.2f} ang={cmd.angular.z:.2f}"
        )
        self.cmd_vel_pub.publish(cmd)





def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()