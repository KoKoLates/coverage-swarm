import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import deque
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection, QuadMesh

from core import Env
from planner import Path
from functools import reduce

from typing import Set, List, Tuple, Optional


class ObstacleHandler(object):
    def __init__(self, vertices: List[Tuple[float, float]], attenuation: float = 0.5):
        self.vertices = np.array(vertices)
        self.center = np.mean(self.vertices, axis=0)
        self.attenuation = attenuation
    
    def _is_point_inside(self, p: np.ndarray) -> bool:
        x, y = p
        inside = False
        n = len(self.vertices)
        
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            
            if y1 == y2:
                continue
                
            if (y1 > y) != (y2 > y):
                if x < ((x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
        
        return inside
    
    def line_attenuation(
        self, 
        p1: np.ndarray, 
        p2: np.ndarray, 
        num_samples: int = 10
    ) -> float:
        samples_x = np.linspace(p1[0], p2[0], num_samples)
        samples_y = np.linspace(p1[1], p2[1], num_samples)
        
        for i in range(num_samples):
            p = np.array([samples_x[i], samples_y[i]])
            if self._is_point_inside(p):
                return self.attenuation
        
        return 1.0
    
    def _vectorized_point_in_polygon(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        inside = np.zeros_like(X, dtype=bool)
        n = len(self.vertices)
        
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            
            if y1 == y2:
                continue
            
            cond1 = (y1 > Y) != (y2 > Y)
            cond2 = X < ((x2 - x1) * (Y - y1) / (y2 - y1) + x1)
            inside ^= (cond1 & cond2)
        
        return inside
    
    def _find_silhouette_vertices(self, source_pos: np.ndarray) -> Tuple[int, int]:
        s_x, s_y = source_pos
        
        dx = self.vertices[:, 0] - s_x
        dy = self.vertices[:, 1] - s_y
        angles = np.arctan2(dy, dx)
        
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]
        
        angle_diffs = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2*np.pi]]))
        max_gap_idx = np.argmax(angle_diffs)
        
        left_idx = sorted_indices[(max_gap_idx + 1) % len(sorted_indices)]
        right_idx = sorted_indices[max_gap_idx]
        
        return left_idx, right_idx
    
    def create_field_attenuation_mask(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        source_pos: np.ndarray
    ) -> np.ndarray:
        s_x, s_y = source_pos
        is_inside = self._vectorized_point_in_polygon(X, Y)
        idx1, idx2 = self._find_silhouette_vertices(source_pos)
        v1 = self.vertices[idx1]  
        v2 = self.vertices[idx2]
        
        a = v2[1] - v1[1]
        b = -(v2[0] - v1[0])
        c = (v2[0] - v1[0]) * v1[1] - (v2[1] - v1[1]) * v1[0]
        
        source_side = a * s_x + b * s_y + c
        grid_side = a * X + b * Y + c
        on_shadow_side = (source_side * grid_side) < 0

        angle1 = np.arctan2(v1[1] - s_y, v1[0] - s_x)
        angle2 = np.arctan2(v2[1] - s_y, v2[0] - s_x)
        angle1 = (angle1 + 2 * np.pi) % (2 * np.pi)
        angle2 = (angle2 + 2 * np.pi) % (2 * np.pi)
        
        grid_angles = np.arctan2(Y - s_y, X - s_x)
        grid_angles = (grid_angles + 2 * np.pi) % (2 * np.pi)
        
        if angle1 <= angle2:
            in_cone = (grid_angles >= angle1) & (grid_angles <= angle2)
        else:
            in_cone = (grid_angles >= angle1) | (grid_angles <= angle2)
        
        shadow_region = (in_cone & on_shadow_side) | is_inside
        
        mask = np.ones_like(X, dtype=np.float32)
        mask[shadow_region] = self.attenuation
        
        return mask


class SignalVisualizer:
    
    TOWER_RANGE = 10
    TOWER_GAIN  = 10
    ROBOT_RANGE = 4
    ROBOT_GAIN  = 4 
    MAX_SIGNAL_DISPLAY = 10.0
    
    def __init__(self, env: Env, resolution: int = 200):
        self.env = env
        self.w, self.h = env.shape
        self.station_position = np.array([self.w // 2, self.h // 2])
        
        # Create heatmap grid
        xs = np.linspace(0, self.w, resolution)
        ys = np.linspace(0, self.h, resolution)
        self.X, self.Y = np.meshgrid(xs, ys)
        
        self.obstacle_handlers: List[ObstacleHandler] = [
            ObstacleHandler(obs, attenuation=0.5) 
            for obs in env.obstacles
        ]
        self.heatmap: QuadMesh = None
        self.robot_plots: List[Line2D] = []
        self.station_plot: PatchCollection = None

    def plot(
        self,
        paths: List[Path],
        file_name: str = "final.png",
        show_robots: bool = True
    ) -> None:
        fig, ax = self._prepare_ax()

        robot_positions = [
            (path[-1].x, path[-1].y)
            for path in paths
        ]

        if show_robots:
            robot_colors = cm.get_cmap("tab20", len(paths))
            for i, pos in enumerate(robot_positions):
                ax.plot(
                    pos[0], pos[1], 
                    marker='o', markersize=8, 
                    color=robot_colors(i)
                )

        Z = self._compute_signal(robot_positions)
        self.heatmap.set_array(Z.ravel())

        if not file_name.endswith(".png"):
            file_name = file_name + ".png"

        plt.savefig(file_name, dpi=200)
        print(f"[INFO] Last frame saved to {file_name}")
        plt.close(fig)

    def animate(
        self, 
        paths: List[Path], 
        file_name: Optional[str] = None, 
        interval: int = 200, 
        show_robots: bool = True
    ) -> None:
        fig, ax = self._prepare_ax()
        
        self.robot_plots: List[Line2D] = []
        if show_robots:
            robot_colors = cm.get_cmap("tab20", len(paths))
            for i in range(len(paths)):
                plot, = ax.plot([], [], marker='o', markersize=8, color=robot_colors(i))
                self.robot_plots.append(plot)
        
        fig.colorbar(self.heatmap, ax=ax, label="Signal Strength")
        
        num_frames = max(len(p) for p in paths)
        
        def update(frame: int):
            robot_positions = [
                (path[min(frame, len(path) - 1)].x, 
                 path[min(frame, len(path) - 1)].y)
                for path in paths
            ]
            
            if show_robots:
                for i, pos in enumerate(robot_positions):
                    if i < len(self.robot_plots):
                        self.robot_plots[i].set_data([pos[0]], [pos[1]])
            
            Z = self._compute_signal(robot_positions)
            self.heatmap.set_array(Z.ravel())
            
            return self.robot_plots + [self.heatmap, self.station_plot]
        
        ani = FuncAnimation(
            fig, update,
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        if file_name is not None:
            ani.save(file_name, writer='pillow', fps=int(1000 / interval), dpi=150)
            print(f"[INFO] Animation saved to {file_name}")
        
        plt.show()
    
    def _gaussian_field(
        self, 
        src_pos: np.ndarray, 
        signal_range: float, 
        gain: float
    ) -> np.ndarray:
        dx = self.X - src_pos[0]
        dy = self.Y - src_pos[1]
        dist_sq = dx * dx + dy * dy
        return gain * np.exp(-dist_sq / (2 * signal_range * signal_range))
    
    def _link_strength(
        self, 
        pos_a: np.ndarray, 
        pos_b: np.ndarray, 
        range_a: float, 
        gain_a: float
    ) -> float:
        dist = np.linalg.norm(pos_a - pos_b)
        strength = gain_a * np.exp(-dist**2 / (2 * range_a**2))

        final_attenuation = reduce(
            lambda acc, handler: acc * handler.line_attenuation(pos_a, pos_b),
            self.obstacle_handlers,
            1.0
        )
        return strength * final_attenuation
    
    def _build_connectivity(self, positions: List[np.ndarray], ranges: List[float]) -> dict:
        n = len(positions)
        adj = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                max_connection_dist = ranges[i] + ranges[j]
                
                if dist <= max_connection_dist:
                    final_attenuation = 1.0
                    for obs_handler in self.obstacle_handlers:
                        final_attenuation *= obs_handler.line_attenuation(
                            positions[i], positions[j]
                        )
                    
                    if final_attenuation > 0.2:
                        adj[i].add(j)
                        adj[j].add(i)
        
        return adj
    
    def _bfs(self, adj: dict, start: int) -> Set[int]:
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited
    
    def _get_connected_robots(self, robot_positions: List[np.ndarray]) -> Set[int]:
        if not robot_positions:
            return set()
        
        positions = [self.station_position] + robot_positions
        ranges = [self.TOWER_RANGE] + [self.ROBOT_RANGE] * len(robot_positions)
        
        adj = self._build_connectivity(positions, ranges)
        visited = self._bfs(adj, start=0)
        
        return {i - 1 for i in visited if i > 0}
    
    def _apply_obstacle_shadows(
        self, 
        field: np.ndarray, 
        src_pos: np.ndarray
    ) -> np.ndarray:
        for obs_handler in self.obstacle_handlers:
            shadow_mask = obs_handler.create_field_attenuation_mask(
                self.X, self.Y, src_pos
            )
            field *= shadow_mask
        return field
    
    def _compute_signal(self, robot_positions: List[Tuple[float, float]]) -> np.ndarray:
        robot_pos_arrays = [np.array(p) for p in robot_positions]
        connected_indices = self._get_connected_robots(robot_pos_arrays)
        
        Z_total = self._apply_obstacle_shadows(
            self._gaussian_field(self.station_position, self.TOWER_RANGE, self.TOWER_GAIN),
            self.station_position
        )
        
        for idx, pos in enumerate(robot_pos_arrays):
            if idx in connected_indices:
                Z_robot = self._apply_obstacle_shadows(
                    self._gaussian_field(pos, self.ROBOT_RANGE, self.ROBOT_GAIN),
                    pos
                )
                Z_total += Z_robot
        
        return np.clip(Z_total, 0, self.MAX_SIGNAL_DISPLAY)
    
    def _prepare_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(7, 7))
        
        for poly in self.env.obstacles:
            ax.add_patch(Polygon(poly, closed=True, facecolor='gray', alpha=0.6, zorder=8))
        
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect("equal")
        ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
        
        Z_init = self._compute_signal([])
        self.heatmap = ax.pcolormesh(
            self.X, self.Y, Z_init,
            cmap='inferno', vmin=0, vmax=self.MAX_SIGNAL_DISPLAY, 
            shading='gouraud'
        )
        
        self.station_plot = ax.scatter(
            self.station_position[0], self.station_position[1],
            marker='x', color='black', s=150, linewidths=2, zorder=10
        )
        return fig, ax
    
    