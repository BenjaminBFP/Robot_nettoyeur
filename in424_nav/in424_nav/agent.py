__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.1.0"

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
from collections import deque
import numpy as np

from .my_common import *

class Agent(Node):
    def __init__(self):
        Node.__init__(self, "Agent")
        self.load_params()
        
        # Attributs
        self.agents_pose = [None] * self.nb_agents
        self.x = self.y = self.yaw = 0.0
        self.ranges = None
        self.current_target = None 
        
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1)
        self.init_map()
        
        self.obstacle_counts = np.zeros(shape=(self.h, self.w), dtype=np.int16)
        
        # Abonnements odométrie optimisés
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", self._make_odom_cb(i-1), 1)
        
        if self.nb_agents != 1:
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data)
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)

        # Timers : Fréquences ajustées pour fluidité/charge
        self.create_timer(0.2, self.map_update) 
        self.create_timer(0.3, self.strategy)   # 3.3 Hz
        self.create_timer(1.0, self.publish_maps)

    def _make_odom_cb(self, idx):
        def cb(msg):
            x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
            if int(self.ns[-1]) == (idx + 1):
                self.x, self.y = x, y
                q = msg.pose.pose.orientation
                self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.agents_pose[idx] = (x, y)
        return cb

    def load_params(self):
        self.declare_parameters(namespace="", parameters=[
            ("ns", rclpy.Parameter.Type.STRING),
            ("robot_size", rclpy.Parameter.Type.DOUBLE),
            ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),
            ("nb_agents", rclpy.Parameter.Type.INTEGER),
        ])
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value

    def init_map(self):
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        res = self.map_msg.info.resolution = self.robot_size
        self.w = self.map_msg.info.width = int(self.env_size[1]/res)
        self.h = self.map_msg.info.height = int(self.env_size[0]/res)
        self.map_msg.info.origin.position.x = -self.env_size[1]/2.0
        self.map_msg.info.origin.position.y = -self.env_size[0]/2.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map = np.ones(shape=(self.h, self.w), dtype=np.int8) * UNEXPLORED_SPACE_VALUE

    def merged_map_cb(self, msg):
        """ FONCTION INCHANGÉE SELON CONSIGNE """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                self.map[i, j] = received_map[i, j]

    def map_update(self):
        """ FONCTION INCHANGÉE SELON CONSIGNE """
        if self.ranges is None or self.x is None: return
        xp_m = []; yp_m = []
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
        grid_size_x, grid_size_y = self.w, self.h

        origin_x, origin_y = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
        robot_i, robot_j = int(np.floor((self.x - origin_x) / resolution)), int(np.floor((-self.y - origin_y) / resolution))
        
        self.map[robot_j, robot_i] = FREE_SPACE_VALUE
        self.obstacle_counts[robot_j, robot_i] = 0

        for r, x, y, detected in zip(self.ranges, xp_m, yp_m, detected_points):                
                i, j = int((x - origin_x) / resolution), int((y - origin_y) / resolution)
                if not (0 <= i < grid_size_x and 0 <= j < grid_size_y): continue
                num = max(abs(i - robot_i), abs(j - robot_j))
                if num == 0: continue
                for k in range(num):
                    xi, yj = int(robot_i + (i - robot_i) * k / num), int(robot_j + (j - robot_j) * k / num)
                    if 0 <= xi < grid_size_x and 0 <= yj < grid_size_y:
                        if self.obstacle_counts[yj, xi] < 4: self.map[yj, xi] = FREE_SPACE_VALUE
                if detected == 1:
                    if self.obstacle_counts[j, i] <= 4: self.obstacle_counts[j, i] += 1
                    self.map[j, i] = OBSTACLE_VALUE

    def lidar_cb(self, msg):
        self.ranges = list(msg.ranges)
        self.angle_increment, self.angle_max, self.angle_min, self.range_max = msg.angle_increment, msg.angle_max, msg.angle_min, msg.range_max

    def publish_maps(self):
        self.map_msg.data = np.flipud(self.map).flatten().tolist()
        self.map_agent_pub.publish(self.map_msg)

    # --- OPTIMISATION : GESTION DE TRAJECTOIRE ET COLLISIONS ---

    def get_distance_map(self, start_i, start_j):
        """ Calcule les distances de TOUTES les cellules au robot en 1 seul passage (BFS optimisé) """
        dist_grid = np.full((self.h, self.w), 999, dtype=np.int16)
        parent_grid = {} 
        if not (0 <= start_i < self.w and 0 <= start_j < self.h): return dist_grid, parent_grid
        
        queue = deque([(start_i, start_j)])
        dist_grid[start_j, start_i] = 0
        
        # Masque des zones interdites (murs + inflation 1 case)
        # On utilise le fait que map == OBSTACLE_VALUE
        is_obstacle = (self.map == OBSTACLE_VALUE)
        
        while queue:
            ci, cj = queue.popleft()
            d = dist_grid[cj, ci]
            if d > 40: break # Limite de recherche pour sauver du CPU
            
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]: # 4-voisinage pour rapidité
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.w and 0 <= nj < self.h:
                    if dist_grid[nj, ni] == 999 and self.map[nj, ni] != OBSTACLE_VALUE:
                        dist_grid[nj, ni] = d + 1
                        parent_grid[(ni, nj)] = (ci, cj)
                        queue.append((ni, nj))
        return dist_grid, parent_grid

    def strategy(self):
        if self.x is None or self.ranges is None: return
        res = self.map_msg.info.resolution
        ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
        ri, rj = int((self.x - ox) / res), int((-self.y - oy) / res)
        msg = Twist()

        # 1. DÉTECTION COLLISION IMMINENTE (Ultra-réactif)
        # On regarde uniquement si un mur est devant nous ET si on avance vers lui
        mid = len(self.ranges) // 2
        front_dist = np.min(np.array(self.ranges)[mid-20 : mid+20])
        if front_dist < 1.5: # Trop proche d'un mur
            self.current_target = None # Invalidation cible
            msg.linear.x = -0.1 # Recul lent
            msg.angular.z = 0.8 # Rotation rapide pour changer d'angle
            self.cmd_vel_pub.publish(msg)
            return

        # 2. CALCUL UNIQUE DES DISTANCES (L'optimisation majeure)
        dist_map, parents = self.get_distance_map(ri, rj)

        # 3. RECHERCHE DE FRONTIERES
        # Vectorisé pour la vitesse
        is_unex = (self.map == UNEXPLORED_SPACE_VALUE)
        is_free = (self.map == FREE_SPACE_VALUE)
        # Une cellule est frontière si Inconnue et touche une cellule Libre
        is_frontier = is_unex & (
            (np.roll(is_free, 1, 0)) | (np.roll(is_free, -1, 0)) | 
            (np.roll(is_free, 1, 1)) | (np.roll(is_free, -1, 1))
        )
        yf, xf = np.where(is_frontier)
        
        if len(xf) == 0:
            msg.angular.z = 0.5; self.cmd_vel_pub.publish(msg); return

        # 4. SCORING DES FRONTIÈRES
        if self.current_target is None or self.map[self.current_target[1], self.current_target[0]] != UNEXPLORED_SPACE_VALUE:
            best_score = -9999
            my_id = int(self.ns[-1]) - 1 # Mon index (0, 1 ou 2)

            # On ne teste que les frontières accessibles (distance < 999)
            for i in range(len(xf)):
                fi, fj = xf[i], yf[i]
                d = dist_map[fj, fi]
                if d == 999: continue
                
                # NOUVEL AJOUT : Distance au robot coéquipier le plus proche
                min_d_others = 1000
                for r_idx, pose in enumerate(self.agents_pose):
                    if pose and r_idx != my_id:
                        oi, oj = int((pose[0]-ox)/res), int((-pose[1]-oy)/res)
                        dist_other = abs(fi - oi) + abs(fj - oj)
                        if dist_other < min_d_others: 
                            min_d_others = dist_other
                
                # Calcul de l'angle diff vers la frontière
                angle_to = np.arctan2(-(fj*res+oy) - self.y, fi*res+ox - self.x)
                angle_diff = abs(np.arctan2(np.sin(angle_to-self.yaw), np.cos(angle_to-self.yaw)))
                
                # FORMULE AVEC COORDINATION : Moi (-) | Angle (-) | Autres (+)
                score = - (d * 4.0) - (angle_diff * 5.0) + (min_d_others * 7.0)

                if score > best_score:
                    best_score = score
                    self.current_target = (fi, fj)

        # 5. NAVIGATION (Reconstruction du chemin via le dictionnaire parents)
        if self.current_target and self.current_target in parents:
            # On remonte le chemin pour trouver un waypoint à 3 cases devant
            path = []
            curr = self.current_target
            while curr in parents:
                path.append(curr)
                curr = parents[curr]
            
            if path:
                target_wp = path[-min(len(path), 3)] # Waypoint local
                tx, ty = target_wp[0]*res+ox, -(target_wp[1]*res+oy)
                
                diff = np.arctan2(ty - self.y, tx - self.x) - self.yaw
                diff = np.arctan2(np.sin(diff), np.cos(diff))
                
                # Commande de vitesse fluide
                msg.linear.x = 0.4 * (1.0 - abs(diff/1.5))
                msg.angular.z = 0.7 * diff
                self.cmd_vel_pub.publish(msg)
                return

        # Fallback
        msg.angular.z = 0.5
        self.cmd_vel_pub.publish(msg)

def main():
    rclpy.init()
    node = Agent()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()