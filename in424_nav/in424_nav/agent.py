__author__ = "Johvany Gustave, Jonatan Alvarez"
__copyright__ = "Copyright 2025, IN424, IPSA 2025"
__credits__ = ["Johvany Gustave", "Jonatan Alvarez"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

#coucou

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

import numpy as np

from .my_common import *    # common variables are stored here


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")

        self.load_params()

        # initialize attributes
        self.agents_pose = [None] * self.nb_agents
        self.x = self.y = self.yaw = None

        # LiDAR data
        self.ranges = None
        self.valid_ranges = None
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0

        # Sector distances for navigation
        self.front_dist = np.inf
        self.left_dist = np.inf
        self.right_dist = np.inf
        self.front_left_dist = np.inf
        self.front_right_dist = np.inf

        # Closest obstacle in all scan
        self.min_obstacle_dist = np.inf
        self.min_obstacle_angle = None

        # Nearby robots / dynamic obstacles
        self.other_agents_detected = []
        self.closest_agent_dist = np.inf
        self.closest_agent_id = None
        self.closest_agent_angle = None

        # Recovery / simple state
        self.last_positions = []
        self.recovery_steps = 0
        self.spin_steps = 10

        self.map_agent_pub = self.create_publisher(
            OccupancyGrid, f"/{self.ns}/map", 1
        )
        self.init_map()

        # Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):
            self.create_subscription(
                Odometry, f"/bot_{i}/odom", odom_methods_cb[i - 1], 1
            )

        if self.nb_agents != 1:
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)

        self.create_subscription(
            LaserScan,
            f"/{self.ns}/laser/scan",
            self.lidar_cb,
            qos_profile=qos_profile_sensor_data
        )

        self.cmd_vel_pub = self.create_publisher(Twist, f"/{self.ns}/cmd_vel", 1)

        # Create timers
        self.create_timer(0.2, self.map_update)    # 5 Hz
        self.create_timer(0.5, self.strategy)      # 2 Hz
        self.create_timer(1.0, self.publish_maps)  # 1 Hz

    def load_params(self):
        """ Load parameters from launch file """
        self.declare_parameters(
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),
                ("robot_size", rclpy.Parameter.Type.DOUBLE),
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),
                ("nb_agents", rclpy.Parameter.Type.INTEGER),
            ]
        )

        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value

    def init_map(self):
        """ Initialize the local occupancy grid """
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.info.resolution = self.robot_size
        self.map_msg.info.height = int(self.env_size[0] / self.map_msg.info.resolution)
        self.map_msg.info.width = int(self.env_size[1] / self.map_msg.info.resolution)
        self.map_msg.info.origin.position.x = -self.env_size[1] / 2
        self.map_msg.info.origin.position.y = -self.env_size[0] / 2
        self.map_msg.info.origin.orientation.w = 1.0

        self.map = np.ones(
            shape=(self.map_msg.info.height, self.map_msg.info.width),
            dtype=np.int8
        ) * UNEXPLORED_SPACE_VALUE

        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height

    def merged_map_cb(self, msg):
        """
        Get the current common map and update ours accordingly.
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                    self.map[i, j] = received_map[i, j]

    def odom1_cb(self, msg):
        """
        @brief Get agent 1 position.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])[2]
        self.agents_pose[0] = (x, y)

    def odom2_cb(self, msg):
        """
        @brief Get agent 2 position.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])[2]
        self.agents_pose[1] = (x, y)

    def odom3_cb(self, msg):
        """
        @brief Get agent 3 position.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])[2]
        self.agents_pose[2] = (x, y)

    def angle_wrap(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def distance_to_other_agents(self):
        """
        Compute the closest other agent using odometry positions.
        Returns (closest_id, distance, relative_angle).
        relative_angle is expressed in robot frame.
        """
        if self.x is None or self.y is None or self.yaw is None:
            return None, np.inf, None

        my_id = int(self.ns[-1])
        best_id = None
        best_dist = np.inf
        best_angle = None

        for i, pose in enumerate(self.agents_pose, start=1):
            if i == my_id or pose is None:
                continue

            ox, oy = pose
            dx = ox - self.x
            dy = oy - self.y
            d = np.hypot(dx, dy)

            if d < best_dist:
                best_dist = d
                best_id = i
                angle_world = np.arctan2(dy, dx)
                rel_angle = self.angle_wrap(angle_world - self.yaw)
                best_angle = rel_angle

        return best_id, best_dist, best_angle

    def get_directional_clearance(self, target_angle, half_width_deg=12):
        """
        Return a robust obstacle distance in a given direction (robot frame),
        based on LiDAR filtered ranges.
        """
        if self.valid_ranges is None or len(self.valid_ranges) == 0:
            return np.inf

        n = len(self.valid_ranges)
        center_idx = int(round((target_angle - self.angle_min) / self.angle_increment)) % n
        half_width = max(1, int(np.deg2rad(half_width_deg) / self.angle_increment))

        vals = []
        for k in range(center_idx - half_width, center_idx + half_width + 1):
            idx = k % n
            r = self.valid_ranges[idx]
            if np.isfinite(r):
                vals.append(float(r))

        if len(vals) == 0:
            return np.inf

        vals = np.array(vals, dtype=np.float32)
        return float(np.percentile(vals, 25))

    def map_update(self):
        """Consider sensor readings to update the agent's map."""

        if self.x is None or self.y is None or self.yaw is None:
            return
        if self.valid_ranges is None:
            return

        resolution = self.map_msg.info.resolution
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y

        def world_to_grid(x, y):
            gx = int(np.floor((x - origin_x) / resolution))
            gy = int(np.floor((y - origin_y) / resolution))
            return gx, gy

        def in_bounds(gx, gy):
            return 0 <= gx < self.w and 0 <= gy < self.h

        def bresenham(x0, y0, x1, y1):
            cells = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = 1 if x1 >= x0 else -1
            sy = 1 if y1 >= y0 else -1

            if dx > dy:
                err = dx / 2
                while x != x1:
                    cells.append((x, y))
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2
                while y != y1:
                    cells.append((x, y))
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy

            cells.append((x1, y1))
            return cells

        def classify_hit(i, r):
            """
            Returns:
                "wall"   -> reliable planar obstacle
                "corner" -> likely real corner / edge
                "none"   -> do not mark obstacle
            """
            if not np.isfinite(r):
                return "none"

            if r >= self.range_max * 0.995:
                return "none"

            n = len(self.valid_ranges)
            prev_r = self.valid_ranges[(i - 1) % n]
            next_r = self.valid_ranges[(i + 1) % n]

            neighbours = []
            if np.isfinite(prev_r):
                neighbours.append(prev_r)
            if np.isfinite(next_r):
                neighbours.append(next_r)

            if r < 1.2 * self.robot_size:
                return "corner"

            if len(neighbours) == 0:
                return "none"

            tolerance_wall = max(0.20, 0.80 * resolution)
            consistent_count = sum(abs(rn - r) <= tolerance_wall for rn in neighbours)

            if consistent_count >= 1:
                return "wall"

            if len(neighbours) == 2:
                d1 = abs(neighbours[0] - r)
                d2 = abs(neighbours[1] - r)

                if d1 > tolerance_wall and d2 > tolerance_wall and r < 3.0 * self.robot_size:
                    return "corner"

            return "none"

        robot_gx, robot_gy = world_to_grid(self.x, self.y)
        if not in_bounds(robot_gx, robot_gy):
            return

        self.map[robot_gy, robot_gx] = FREE_SPACE_VALUE

        step = max(1, len(self.valid_ranges) // 240)

        for i in range(0, len(self.valid_ranges), step):
            r = self.valid_ranges[i]

            if not np.isfinite(r):
                continue

            hit_type = classify_hit(i, r)
            hit_obstacle = hit_type in ["wall", "corner"]

            used_r = r if hit_obstacle else min(r, self.range_max * 0.98)

            beam_angle = self.yaw + self.angle_min + i * self.angle_increment

            end_x = self.x + used_r * math.cos(beam_angle)
            end_y = self.y + used_r * math.sin(beam_angle)

            end_gx, end_gy = world_to_grid(end_x, end_y)
            if not in_bounds(end_gx, end_gy):
                continue

            ray_cells = bresenham(robot_gx, robot_gy, end_gx, end_gy)

            free_cells = ray_cells[:-1] if hit_obstacle else ray_cells
            for cx, cy in free_cells:
                if in_bounds(cx, cy) and self.map[cy, cx] != OBSTACLE_VALUE:
                    self.map[cy, cx] = FREE_SPACE_VALUE

            if hit_obstacle and in_bounds(end_gx, end_gy):
                self.map[end_gy, end_gx] = OBSTACLE_VALUE

                if hit_type == "corner":
                    for nx, ny in [
                        (end_gx + 1, end_gy),
                        (end_gx - 1, end_gy),
                        (end_gx, end_gy + 1),
                        (end_gx, end_gy - 1)
                    ]:
                        if in_bounds(nx, ny) and self.map[ny, nx] == UNEXPLORED_SPACE_VALUE:
                            self.map[ny, nx] = FREE_SPACE_VALUE

    def lidar_cb(self, msg):
        """
        @brief Get messages from LIDAR topic.
        This method is automatically called whenever a new message is published on topic /bot_x/laser/scan.
        """

        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        if raw_ranges.size == 0:
            self.ranges = None
            self.valid_ranges = None
            self.front_dist = np.inf
            self.left_dist = np.inf
            self.right_dist = np.inf
            self.front_left_dist = np.inf
            self.front_right_dist = np.inf
            self.min_obstacle_dist = np.inf
            self.min_obstacle_angle = None
            self.other_agents_detected = []
            self.closest_agent_dist = np.inf
            self.closest_agent_id = None
            self.closest_agent_angle = None
            return

        self.ranges = raw_ranges.copy()
        self.valid_ranges = raw_ranges.copy()

        invalid_mask = (
            ~np.isfinite(self.valid_ranges) |
            (self.valid_ranges < self.range_min) |
            (self.valid_ranges > self.range_max)
        )
        self.valid_ranges[invalid_mask] = np.inf

        n = len(self.valid_ranges)

        def angle_to_index(angle):
            idx = int(round((angle - self.angle_min) / self.angle_increment))
            return idx % n

        def sector_distance(center_angle, half_width_deg=15):
            center_idx = angle_to_index(center_angle)
            half_width = max(1, int(np.deg2rad(half_width_deg) / self.angle_increment))

            vals = []
            for k in range(center_idx - half_width, center_idx + half_width + 1):
                idx = k % n
                r = self.valid_ranges[idx]
                if np.isfinite(r):
                    vals.append(float(r))

            if len(vals) == 0:
                return np.inf

            vals = np.array(vals, dtype=np.float32)
            return float(np.percentile(vals, 25))

        self.front_dist = sector_distance(0.0, 18)
        self.left_dist = sector_distance(np.pi / 2, 20)
        self.right_dist = sector_distance(-np.pi / 2, 20)
        self.front_left_dist = sector_distance(np.pi / 4, 15)
        self.front_right_dist = sector_distance(-np.pi / 4, 15)

        finite_ids = np.where(np.isfinite(self.valid_ranges))[0]
        if len(finite_ids) > 0:
            best_idx = finite_ids[np.argmin(self.valid_ranges[finite_ids])]
            self.min_obstacle_dist = float(self.valid_ranges[best_idx])
            self.min_obstacle_angle = self.angle_min + best_idx * self.angle_increment
        else:
            self.min_obstacle_dist = np.inf
            self.min_obstacle_angle = None

        self.other_agents_detected = []
        self.closest_agent_dist = np.inf

        max_gap = 1
        candidate_clusters = []

        front_ids = []
        for i in range(n):
            angle = self.angle_min + i * self.angle_increment
            if -np.pi / 2 <= angle <= np.pi / 2 and np.isfinite(self.valid_ranges[i]):
                front_ids.append(i)

        close_ids = [i for i in front_ids if self.valid_ranges[i] <= max(1.5, 3.0 * self.robot_size)]

        if len(close_ids) > 0:
            cluster = [close_ids[0]]
            for idx in close_ids[1:]:
                if idx - cluster[-1] <= max_gap:
                    cluster.append(idx)
                else:
                    candidate_clusters.append(cluster)
                    cluster = [idx]
            candidate_clusters.append(cluster)

        for cluster_ids in candidate_clusters:
            if len(cluster_ids) < 3:
                continue

            distances = np.array([self.valid_ranges[i] for i in cluster_ids], dtype=np.float32)
            mean_r = float(np.mean(distances))
            min_r = float(np.min(distances))

            ang0 = self.angle_min + cluster_ids[0] * self.angle_increment
            ang1 = self.angle_min + cluster_ids[-1] * self.angle_increment
            ang_span = abs(ang1 - ang0)
            observed_width = mean_r * ang_span

            if 0.4 * self.robot_size <= observed_width <= 2.2 * self.robot_size:
                center_idx = cluster_ids[len(cluster_ids) // 2]
                center_angle = self.angle_min + center_idx * self.angle_increment

                self.other_agents_detected.append({
                    "distance": mean_r,
                    "min_distance": min_r,
                    "angle": center_angle,
                    "width": observed_width,
                    "indices": cluster_ids
                })
                self.closest_agent_dist = min(self.closest_agent_dist, min_r)

    def publish_maps(self):
        """
        Publish updated map to topic /bot_x/map.
        """
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.data = np.flipud(self.map).flatten().tolist()
        self.map_agent_pub.publish(self.map_msg)

    def strategy(self):
        """Decision and action layers based on obstacle and inter-robot distances."""
        cmd = Twist()

        if self.x is None or self.y is None or self.yaw is None:
            self.cmd_vel_pub.publish(cmd)
            return

        if self.valid_ranges is None:
            self.cmd_vel_pub.publish(cmd)
            return

        robot_id = int(self.ns[-1])

        if self.spin_steps > 0:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.7
            self.spin_steps -= 1
            self.cmd_vel_pub.publish(cmd)
            return

        self.closest_agent_id, self.closest_agent_dist, self.closest_agent_angle = self.distance_to_other_agents()

        self.last_positions.append((self.x, self.y))
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)

        if len(self.last_positions) >= 10:
            x0, y0 = self.last_positions[0]
            moved = np.hypot(self.x - x0, self.y - y0)

            if moved < 0.05 and self.front_dist < 1.2:
                self.recovery_steps = 6

        if self.recovery_steps > 0:
            cmd.linear.x = -0.03
            cmd.angular.z = 0.8 if self.left_dist > self.right_dist else -0.8
            self.recovery_steps -= 1
            self.cmd_vel_pub.publish(cmd)
            return

        FRONT_STOP = max(0.75, 1.4 * self.robot_size)
        FRONT_SLOW = max(1.20, 2.2 * self.robot_size)
        SIDE_SAFE = max(0.55, 1.5 * self.robot_size)
        AGENT_SAFE = max(0.90, 2.0 * self.robot_size)

        if self.closest_agent_dist < AGENT_SAFE and self.closest_agent_angle is not None:
            if abs(self.closest_agent_angle) < np.pi / 2:
                cmd.linear.x = 0.0

                if self.closest_agent_angle >= 0:
                    cmd.angular.z = -0.7 if robot_id != 2 else 0.7
                else:
                    cmd.angular.z = 0.7 if robot_id != 2 else -0.7

                self.cmd_vel_pub.publish(cmd)
                return

        if self.front_dist < FRONT_STOP:
            cmd.linear.x = 0.0

            left_clearance = min(self.left_dist, self.front_left_dist)
            right_clearance = min(self.right_dist, self.front_right_dist)

            if left_clearance > right_clearance:
                cmd.angular.z = 0.8
            elif right_clearance > left_clearance:
                cmd.angular.z = -0.8
            else:
                cmd.angular.z = 0.8 if robot_id in [1, 3] else -0.8

            self.cmd_vel_pub.publish(cmd)
            return

        if self.front_dist > 2.0:
            cmd.linear.x = 0.16
        elif self.front_dist > FRONT_SLOW:
            cmd.linear.x = 0.11
        else:
            cmd.linear.x = 0.06

        wall_error = 0.0
        if np.isfinite(self.left_dist) and np.isfinite(self.right_dist):
            wall_error = self.right_dist - self.left_dist
            wall_error = np.clip(wall_error, -0.6, 0.6)

        wall_correction = 0.8 * wall_error

        side_bias = 0.0
        if self.left_dist < SIDE_SAFE:
            side_bias -= 0.25
        if self.right_dist < SIDE_SAFE:
            side_bias += 0.25

        exploration_bias = 0.0
        if robot_id == 1:
            exploration_bias = 0.08
        elif robot_id == 2:
            exploration_bias = -0.08

        cmd.angular.z = wall_correction + side_bias + exploration_bias
        cmd.angular.z = float(np.clip(cmd.angular.z, -0.8, 0.8))

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
