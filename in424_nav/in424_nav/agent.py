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
                i = int((round((x - origin_x) / resolution)))
                j = int(round(((y - origin_y) / resolution)))

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


    def get_frontiers(self, x_min, x_max):
        """
        Parcourt toutes les cases de la carte entre x_min et x_max
        et retourne les cases inexplorées qui ont au moins un voisin libre.
        Ce sont les "frontières" : la limite entre connu et inconnu.
        """
        frontiers = []
        for j in range(self.h):
            for i in range(x_min, x_max):
                if self.map[j, i] == UNEXPLORED_SPACE_VALUE:
                    # Vérifier les 4 voisins (haut, bas, gauche, droite)
                    neighbors = [(j-1, i), (j+1, i), (j, i-1), (j, i+1)]
                    for nj, ni in neighbors:
                        if 0 <= nj < self.h and 0 <= ni < self.w:
                            if self.map[nj, ni] == FREE_SPACE_VALUE:
                                frontiers.append((i, j))
                                break  # pas besoin de vérifier les autres voisins
        return frontiers
    
    
    def strategy(self):
        if self.x is None or self.map is None or self.ranges is None:
            return

        # -------------------------------------------------------
        # SÉCURITÉ — Vérifier si un obstacle est trop proche
        # On regarde les rayons LIDAR dans un cône devant le robot
        # -------------------------------------------------------
        SAFETY_DISTANCE = 1 # distance de sécurité en mètres

        ranges = np.array(self.ranges)
        n = len(ranges)
        angles = np.linspace(self.angle_min, self.angle_max, n, endpoint=False)

        # On ne regarde que les rayons dans un cône de ±45° devant le robot
        # (angles entre -pi/4 et pi/4 dans le repère du robot)
        front_mask = (angles > -np.pi/4) & (angles < np.pi/4)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        # Remplacer les inf par range_max pour le calcul
        front_ranges = np.where(np.isinf(front_ranges), self.range_max, front_ranges)

        # Y a-t-il un obstacle dans le cône avant à moins de SAFETY_DISTANCE ?
        obstacle_ahead = np.any(front_ranges < SAFETY_DISTANCE)

        if obstacle_ahead:
            # Calculer de quel côté l'obstacle est le plus proche
            # pour tourner dans la direction opposée
            left_mask = front_angles > 0
            right_mask = front_angles < 0

            min_left = np.min(front_ranges[left_mask]) if np.any(left_mask) else self.range_max
            min_right = np.min(front_ranges[right_mask]) if np.any(right_mask) else self.range_max

            msg = Twist()
            msg.linear.x = 0.0  # on arrête d'avancer

            if min_left < min_right:
                # obstacle plus proche à gauche → tourner à droite
                msg.angular.z = -0.5
            else:
                # obstacle plus proche à droite → tourner à gauche
                msg.angular.z = 0.5

            self.cmd_vel_pub.publish(msg)
            return  # on ne continue pas vers la frontière tant qu'il y a un obstacle

        # -------------------------------------------------------
        # Si pas d'obstacle → continuer la stratégie frontier
        # -------------------------------------------------------
        robot_id = int(self.ns[-1])
        zone_width = self.w // self.nb_agents
        zone_x_min = (robot_id - 1) * zone_width
        zone_x_max = zone_x_min + zone_width if robot_id < self.nb_agents else self.w

        frontiers = self.get_frontiers(zone_x_min, zone_x_max)
        if len(frontiers) == 0:
            frontiers = self.get_frontiers(0, self.w)
        if len(frontiers) == 0:
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            return

        resolution = self.map_msg.info.resolution
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y

        robot_i = int(round((self.x - origin_x) / resolution))
        robot_j = int(round((-self.y - origin_y) / resolution))

        closest = min(frontiers, key=lambda f: (f[0] - robot_i)**2 + (f[1] - robot_j)**2)

        target_x = closest[0] * resolution + origin_x
        target_y = -(closest[1] * resolution + origin_y)

        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_target - self.yaw), np.cos(angle_to_target - self.yaw))

        msg = Twist()
        if abs(angle_diff) > 0.2:
            msg.linear.x = 0.0
            msg.angular.z = 0.5 * np.sign(angle_diff)
        elif distance > 0.3:
            msg.linear.x = 0.3
            msg.angular.z = 0.2 * angle_diff
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)


def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()