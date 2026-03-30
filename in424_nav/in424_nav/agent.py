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
from time import time
from .my_common import *
import json
class Agent(Node):
    def __init__(self):
        Node.__init__(self, "Agent")
        self.load_params()
        
        # Attributs
        self.current_path = []
        self.path_idx = 0
        self.last_map_hash = None
        self.agents_pose = [None] * self.nb_agents
        self.x = self.y = self.yaw = 0.0
        self.ranges = None
        self.current_target = None 
        self.my_id = int(self.ns[-1]) - 1  # Mon index (0, 1 ou 2)
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1)
        self.obstacle_counts_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/obstacle_counts", 1)
        self.init_map()
        self.obstacle_counts = np.zeros(shape=(self.h, self.w), dtype=np.int16)
        
        # Abonnements odométrie optimisés
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", self._make_odom_cb(i-1), 1)
        
        if self.nb_agents != 1:
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
            self.create_subscription(OccupancyGrid, "/merged_obstacle_counts", self.merged_obstacle_counts_cb, 1)
        
        self.create_subscription(LaserScan, f"{self.ns}/laser/scan", self.lidar_cb, qos_profile=qos_profile_sensor_data)
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)

        # Timers : Fréquences ajustées pour fluidité/charge
        self.create_timer(0.3, self.map_update) 
        self.create_timer(0.22, self.strategy)   # 3.3 Hz
        self.create_timer(1.3, self.publish_maps)

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
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                self.map[i, j] = received_map[i, j]

    def map_update(self):
        """ FONCTION INCHANGÉE SELON CONSIGNE """
        if self.ranges is None or self.x is None: return
        ranges = np.array(self.ranges)
        xp_m = []; yp_m = []
        angles = np.linspace(self.angle_min, self.angle_max, len(self.ranges), endpoint=False)
        yaw = self.yaw
        x = self.x
        y = self.y
        agents_pose = self.agents_pose
        detected_points = []    
        for i, r in enumerate(ranges) :
            if np.isinf(r) :
                detected_points.append(0)
                ranges[i] = self.range_max
                r = self.range_max-1.5
            else :
                detected_points.append(1)
            xp_m.append(r * np.cos(angles[i]+yaw)+ x)
            yp_m.append(-r * np.sin(angles[i]+yaw)- y)
       
        resolution = self.map_msg.info.resolution
        grid_size_x, grid_size_y = self.w, self.h

        origin_x, origin_y = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
        calcul_robot_i, calcul_robot_j = (x - origin_x) / resolution, (-y - origin_y) / resolution
        
        robot_i, robot_j = int(calcul_robot_i), int(calcul_robot_j) 
        self.map[robot_j, robot_i] = FREE_SPACE_VALUE
        self.obstacle_counts[robot_j, robot_i] = 0
        
        for r, x_, y_, detected in zip(ranges, xp_m, yp_m, detected_points):
                calcul_i, calcul_j = (x_ - origin_x) / resolution, (y_ - origin_y) / resolution     
                i, j = int(calcul_i), int(calcul_j)
                if calcul_i<0 :
                    i=0
                    # self.get_logger().info(f"i stoo low: {i}")
                if calcul_i>=grid_size_x :
                    i=grid_size_x-1
                    # self.get_logger().info(f"i too high: {i}")
                if calcul_j<0 :
                    j=0
                    # self.get_logger().info(f"j too low: {j}")
                if calcul_j>=grid_size_y :
                    j=grid_size_y-1        
                    # self.get_logger().info(f"j too high: {j}")
                num = int(max(abs(i - calcul_robot_i), abs(j - calcul_robot_j)))
                if num == 0: continue
                for k in range(num):
                    xi, yj = int(np.round(calcul_robot_i + (calcul_i - calcul_robot_i) * k / num)), int(np.round(calcul_robot_j + (calcul_j - calcul_robot_j) * k / num))
                    if 0 <= xi < grid_size_x and 0 <= yj < grid_size_y and self.obstacle_counts[yj, xi] <2:
                        self.map[yj, xi] = FREE_SPACE_VALUE 
                if detected == 1:
                    is_teammate = False      
                    for r_idx, pose in enumerate(agents_pose):
                        if pose and r_idx != self.my_id:
                            pi = int(np.round((pose[0] - origin_x) / resolution))
                            pj = int(np.round((-pose[1] - origin_y) / resolution))
                            if abs(i - pi) <= 2 and abs(j - pj) <= 2:  # tolérance 3 cases
                                is_teammate = True
                                break
                    if not is_teammate:
                        if self.obstacle_counts[j, i] <= 2:
                            self.obstacle_counts[j, i] += 1
                        self.map[j, i] = OBSTACLE_VALUE
                    else :
                        self.map[j, i] = FREE_SPACE_VALUE
                else:
                    self.obstacle_counts[j, i] -= 1
                    self.map[j, i] = FREE_SPACE_VALUE


    def lidar_cb(self, msg):
        self.ranges = list(msg.ranges)
        self.angle_increment, self.angle_max, self.angle_min, self.range_max = msg.angle_increment, msg.angle_max, msg.angle_min, msg.range_max

    def publish_maps(self):
        self.map_msg.data = np.flipud(self.map).flatten().tolist()
        self.map_agent_pub.publish(self.map_msg)

    def publish_obstacle_counts(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.info = self.map_msg.info
        msg.data = np.flipud(self.obstacle_counts).flatten().tolist()
        self.obstacle_counts_pub.publish(msg)

    def merged_obstacle_counts_cb(self, msg):
        received = np.flipud(np.array(msg.data).reshape(self.h, self.w))
        # Prendre le max entre compteur local et reçu
        self.obstacle_counts = np.maximum(self.obstacle_counts, received).astype(np.int16)

    def calcul_time(self,time_start):
        time_end = time()
        Dt = round(time_end - time_start,3)
        log_time = "./ros2_ws/src/IN424/in424_nav/in424_nav/log_time.json"
        try:
            with open(log_time, "r") as f:
                donnees = json.load(f)
        except:
            donnees = []

            # Ajout
        donnees.append(Dt)

        # Écriture
        with open(log_time, "w") as f:
            json.dump(donnees, f)    
    def path_still_valid(self):
        """Vérifie que aucune cellule du chemin restant n'est devenue un obstacle"""
        for wi, wj in self.current_path[self.path_idx:]:
            if self.map[wj, wi] == OBSTACLE_VALUE:
                return False
        return True
    
    def is_safe(self, i, j):
        # 1 cellule de marge = 0.5m = 1 rayon robot
        margin_cells = 1
        for di in range(-margin_cells, margin_cells + 1):
            for dj in range(-margin_cells, margin_cells + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.w and 0 <= nj < self.h:
                    if self.map[nj, ni] == OBSTACLE_VALUE:
                        return False
        return True

    def find_path_dfs(self, start_i, start_j, goal_i, goal_j, max_depth=200, margin=None, shuffle_seed=None):
        if margin is None:
            margin = 1

        stack = [(start_i, start_j, [(start_i, start_j)])]
        visited = set()
        visited.add((start_i, start_j))

        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        rng = np.random.RandomState(shuffle_seed)  # Graine contrôlée

        while stack:
            ci, cj, path = stack.pop()

            if ci == goal_i and cj == goal_j:
                return path, False

            if len(path) > max_depth:
                continue

            neighbors = []
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.w and 0 <= nj < self.h): continue
                if (ni, nj) in visited: continue

                cell = self.map[nj, ni]

                if cell == UNEXPLORED_SPACE_VALUE:
                    return path + [(ni, nj)], True

                if cell == FREE_SPACE_VALUE and (margin == 0 or self.is_safe(ni, nj)):
                    dist = abs(ni - goal_i) + abs(nj - goal_j)
                    # Ajout d'un bruit aléatoire pour varier les chemins
                    noise = rng.uniform(0, 3)
                    neighbors.append((dist + noise, ni, nj))

            neighbors.sort(reverse=True)
            for _, ni, nj in neighbors:
                visited.add((ni, nj))
                stack.append((ni, nj, path + [(ni, nj)]))

        return None, False


    def find_best_path(self, ri, rj, goal_i, goal_j, margin=None):
        """Lance 5 DFS avec graines différentes, retourne le chemin le plus court"""
        best_path = None
        best_hit_unex = False

        for seed in range(5):
            path, hit_unex = self.find_path_dfs(ri, rj, goal_i, goal_j, margin=margin, shuffle_seed=seed)
            if path is None: continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_hit_unex = hit_unex

        return best_path, best_hit_unex

    def strategy(self):
        time_start = time()
        if self.x is None or self.ranges is None: return

        res = self.map_msg.info.resolution
        ox, oy = self.map_msg.info.origin.position.x, self.map_msg.info.origin.position.y
        ri = int((self.x - ox) / res)
        rj = int((-self.y - oy) / res)
        msg = Twist()
        # self.get_logger().info(
        #     f"pos=({self.x:.2f},{self.y:.2f}) "
        #     f"grid=({ri},{rj}) "
        #     f"cell={self.map[rj,ri] if 0<=ri<self.w and 0<=rj<self.h else 'OUT'} "
        #     f"target={self.current_target} "
        #     f"path_len={len(self.current_path)} "
        #     f"path_idx={self.path_idx} "
        #     f"next_wp={self.current_path[self.path_idx] if self.current_path else None}"
        # )
        # 1. RECHERCHE DE FRONTIÈRES
        is_unex = (self.map == UNEXPLORED_SPACE_VALUE)
        is_free = (self.map == FREE_SPACE_VALUE)
        is_frontier = is_unex & (
            (np.roll(is_free, 1, 0)) | (np.roll(is_free, -1, 0)) |
            (np.roll(is_free, 1, 1)) | (np.roll(is_free, -1, 1))
        )
        yf, xf = np.where(is_frontier)
        if len(xf) == 0:
            msg.angular.z = 0.5
            self.cmd_vel_pub.publish(msg)
            return

        # 2. CHOIX DE CIBLE (seulement si on n'en a pas)
        if (self.current_target is None or
            self.map[self.current_target[1], self.current_target[0]] != UNEXPLORED_SPACE_VALUE):

            self.current_path = []  # Invalider le chemin aussi
            self.path_idx = 0
            best_score = -9999
            my_id = int(self.ns[-1]) - 1
            for idx in range(len(xf)):
                fi, fj = xf[idx], yf[idx]
                d = abs(fi - ri) + abs(fj - rj)
                min_d_others = 1000
                for r_idx, pose in enumerate(self.agents_pose):
                    if pose and r_idx != my_id:
                        oi = int((pose[0]-ox)/res)
                        oj = int((-pose[1]-oy)/res)
                        dist_other = abs(fi-oi) + abs(fj-oj)
                        if dist_other < min_d_others:
                            min_d_others = dist_other
                angle_to = np.arctan2(-(fj*res+oy) - self.y, fi*res+ox - self.x)
                angle_diff = abs(np.arctan2(np.sin(angle_to-self.yaw), np.cos(angle_to-self.yaw)))
                score = -(d * 5.0) - (angle_diff * 6.0) + (min_d_others * 8.0)
                if score > best_score:
                    best_score = score
                    self.current_target = (fi, fj)

        # 3. CALCUL DU CHEMIN (seulement si pas de chemin en cours)
        if not self.current_path or not self.path_still_valid():
            self.current_path = []
            self.path_idx = 0
        if not self.current_path:
            path, hit_unex = self.find_best_path(ri, rj, self.current_target[0], self.current_target[1])
            
            if path is None or len(path) < 2:
                # Fallback : recalcul sans marge de sécurité
                path, hit_unex = self.find_best_path(ri, rj, self.current_target[0], self.current_target[1])
            
            if path is None or len(path) < 2:
                # Vraiment inaccessible → on tourne et on réessaie au prochain cycle
                msg.angular.z = 0.5
                self.cmd_vel_pub.publish(msg)
                self.calcul_time(time_start)
                return
            
            self.current_path = path
            self.path_idx = 0
        # 4. AVANCER DANS LE CHEMIN : passer au waypoint suivant si assez proche
        while self.path_idx < len(self.current_path) - 1:
            wi, wj = self.current_path[self.path_idx]
            tx = wi * res + ox
            ty = -(wj * res + oy)
            dist_to_wp = np.sqrt((self.x - tx)**2 + (self.y - ty)**2)
            if dist_to_wp < res * 1.5:
                self.path_idx += 1
            else:
                break

        # Détection fin de chemin : dernier waypoint atteint OU assez proche
        if self.path_idx >= len(self.current_path) - 1:
            wi, wj = self.current_path[-1]
            tx = wi * res + ox
            ty = -(wj * res + oy)
            dist_final = np.sqrt((self.x - tx)**2 + (self.y - ty)**2)
            if dist_final < res * 3.0:  # Tolérance plus large pour la fin
                self.current_path = []
                self.path_idx = 0
                self.current_target = None  # Recalcul cible + chemin
                msg.angular.z = 0.5
                self.cmd_vel_pub.publish(msg)
                self.calcul_time(time_start)
                return
        # 5. NAVIGATION vers waypoint courant
        wi, wj = self.current_path[self.path_idx]
        tx = wi * res + ox
        ty = -(wj * res + oy)

        diff = np.arctan2(ty - self.y, tx - self.x) - self.yaw
        diff = np.arctan2(np.sin(diff), np.cos(diff))

        msg.linear.x = 0.4 * (1.0 - abs(diff / 1.5))
        msg.angular.z = 0.7 * diff
        self.cmd_vel_pub.publish(msg)
        self.calcul_time(time_start)

def main():
    rclpy.init()
    node = Agent()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()
