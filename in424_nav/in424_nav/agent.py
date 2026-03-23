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
from collections import deque
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
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height
        self.current_target = None  # (i, j) de la frontière cible
        self.next_wp = None        # (i, j) de la prochaine case
        # AJOUT : Matrice pour compter les détections d'obstacles
        self.obstacle_counts = np.zeros(shape=(self.h, self.w), dtype=np.int16)
        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own LIDAR topic
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.1s of period <=> 5 Hz
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
                i = int((x - origin_x) / resolution)
                j = int((y - origin_y) / resolution)

                # hors map
                if not (0 <= i < grid_size_x and 0 <= j < grid_size_y):
                    continue

                # obstacle
                

                # espace libre 
                num = max(abs(i - robot_i), abs(j - robot_j))

                if num == 0:
                    continue

                for k in range(num):

                    xi = int(robot_i + (i - robot_i) * k / num)
                    yj = int(robot_j + (j - robot_j) * k / num)

                    if 0 <= xi < grid_size_x and 0 <= yj < grid_size_y:
                        if self.obstacle_counts[yj, xi] < 4:
                            self.map[yj, xi] = FREE_SPACE_VALUE
                if detected == 1:
                    if self.obstacle_counts[j, i] <= 4: 
                        self.obstacle_counts[j, i] += 1
                    self.map[j, i] = OBSTACLE_VALUE

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


    def is_occupied_by_other_robot(self, i, j, res, ox, oy):
        for pose in self.agents_pose:
            if pose is None: continue
            if abs(pose[0]-self.x) < 0.2 and abs(pose[1]-self.y) < 0.2: continue # ignore soi
            oi, oj = int((pose[0]-ox)/res), int((-pose[1]-oy)/res)
            if abs(i - oi) <= 1 and abs(j - oj) <= 1:
                return True
        return False
    
    def get_frontiers(self):
        frontiers = []
        for j in range(1, self.h - 1):
            for i in range(1, self.w - 1):
                if self.map[j, i] == UNEXPLORED_SPACE_VALUE:
                    # Une frontière est une zone inexplorée adjacente au vide
                    for nj, ni in [(j-1,i),(j+1,i),(j,i-1),(j,i+1)]:
                        if self.map[nj, ni] == FREE_SPACE_VALUE:
                            frontiers.append((i, j))
                            break
        return frontiers

    # --- Pathfinding BFS ---
    def is_cell_safe(self, i, j):
        """Vérifie si une case n'est pas un obstacle et n'est pas trop proche d'un mur (Inflation)"""
        if not (0 <= i < self.w and 0 <= j < self.h): return False
        if self.map[j, i] == OBSTACLE_VALUE: return False
        
        # Inflation manuelle : on vérifie les 8 voisins. Si un mur est à 1 case, on évite.
        for ni, nj in [(i-1,j),(i+1,j),(i,j-1),(i,j+1), (i-1,j-1), (i+1,j+1), (i-1,j+1), (i+1,j-1)]:
            if 0 <= ni < self.w and 0 <= nj < self.h:
                if self.map[nj, ni] == OBSTACLE_VALUE:
                    return False
        return True

    def compute_path(self, start_i, start_j, target_i, target_j, res, ox, oy):
        from collections import deque
        if (start_i, start_j) == (target_i, target_j): return 0, (target_i, target_j)
            
        queue = deque([(start_i, start_j, 0)])
        parent = {(start_i, start_j): None}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while queue:
            ci, cj, dist = queue.popleft()
            if (ci, cj) == (target_i, target_j):
                path = []
                curr = (ci, cj)
                while curr is not None:
                    path.append(curr); curr = parent[curr]
                path.reverse()
                # On vise à 2 cases devant pour ne pas raser les angles de trop près
                idx = 2 if len(path) > 2 else (len(path)-1)
                return dist, path[idx]

            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.w and 0 <= nj < self.h and (ni, nj) not in parent:
                    # RÈGLE : Passer uniquement sur les cases SURES (pas de murs autour)
                    # Sauf si on est au point de départ ou d'arrivée
                    is_special = (ni, nj) == (target_i, target_j) or (ni, nj) == (start_i, start_j)
                    if self.is_cell_safe(ni, nj) or is_special or self.is_occupied_by_other_robot(ni, nj, res, ox, oy):
                        parent[(ni, nj)] = (ci, cj)
                        queue.append((ni, nj, dist + 1))
        return float('inf'), None

    def strategy(self):
        if self.x is None or self.ranges is None: return
        res = self.map_msg.info.resolution
        ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
        ri, rj = int((self.x - ox) / res), int((-self.y - oy) / res)
        msg = Twist()

        # 1. RÉFLEXE D'ÉVITEMENT AVEC ABANDON DE CIBLE
        ranges_clean = np.where(np.isinf(self.ranges), self.range_max, self.ranges)
        mid = len(ranges_clean) // 2
        # On regarde si quelque chose nous bloque vraiment à < 0.8m
        if np.min(ranges_clean[mid-25:mid+25]) < 0.8:
            self.current_target = None  # ABANDON de la cible car bloqué par mur physique
            msg.linear.x = -0.2       # Petit recul pour sortir du piège
            msg.angular.z = 0.8        # Rotation forte
            self.cmd_vel_pub.publish(msg)
            return

        # 2. FRONTIÈRES
        frontiers = self.get_frontiers()
        if not frontiers:
            msg.angular.z = 0.5; self.cmd_vel_pub.publish(msg)
            return

        if self.current_target and self.map[self.current_target[1], self.current_target[0]] != UNEXPLORED_SPACE_VALUE:
            self.current_target = None

        # 3. PRISE DE DÉCISION (Nouveau calcul de Score avec pénalité angulaire)
        if self.current_target is None:
            best_score = -float('inf')
            # Trier par distance euclidienne avant de tester au BFS
            frontiers.sort(key=lambda f: (f[0]-ri)**2 + (f[1]-rj)**2)
            
            for f in frontiers[:40]:
                d_me, _ = self.compute_path(ri, rj, f[0], f[1], res, ox, oy)
                if d_me == float('inf'): continue
                
                # Calcul de l'angle pour atteindre cette frontière
                fx, fy = f[0]*res + ox, -(f[1]*res + oy)
                angle_to_f = np.arctan2(fy - self.y, fx - self.x)
                angle_diff = abs(np.arctan2(np.sin(angle_to_f - self.yaw), np.cos(angle_to_f - self.yaw)))
                
                # Coopération (Distance des autres)
                min_d_others = 1000
                for pose in self.agents_pose:
                    if pose is None or abs(pose[0]-self.x) < 0.2: continue
                    oi, oj = int((pose[0]-ox)/res), int((-pose[1]-oy)/res)
                    min_d_others = min(min_d_others, abs(f[0]-oi) + abs(f[1]-oj))
                
                # --- SCORE RÉVISÉ ---
                # dist_me pénalisé, dist_others récompensé
                # GROSSE PÉNALITÉ si la cible est derrière nous (angle_diff grand)
                score = (min_d_others * 2.0) - (d_me * 6.0) - (angle_diff * 15.0)
                
                if score > best_score:
                    best_score = score
                    self.current_target = f

        # 4. NAVIGATION VERS LE WAYPOINT
        if self.current_target:
            _, wp = self.compute_path(ri, rj, self.current_target[0], self.current_target[1], res, ox, oy)
            if wp:
                tx, ty = wp[0]*res + ox, -(wp[1]*res + oy)
                diff = np.arctan2(ty - self.y, tx - self.x) - self.yaw
                diff = np.arctan2(np.sin(diff), np.cos(diff))
                
                if abs(diff) > 0.6: # Virage important
                    msg.linear.x = 0.2 # On rampe pour mieux pivoter
                    msg.angular.z = 0.8 * np.sign(diff)
                else: # Aligné
                    msg.linear.x = 0.7 * (1.0 - abs(diff)) # Vitesse dégressive pour fluidité
                    msg.angular.z = 0.5 * diff
                
                self.cmd_vel_pub.publish(msg)
                return

        # Fallback
        self.current_target = None
        msg.angular.z = 0.6; self.cmd_vel_pub.publish(msg)
    
def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()