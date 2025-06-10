"""
Improved Ripley K/L Function Analysis Package  -- v0.9 2025-06-09
============================================

A refactored and enhanced version of the spatial point pattern analysis code
using Ripley's K and L functions for cell distribution analysis.

Author: Refactored version

This version includes:
- Improved data preprocessing
- Enhanced boundary correction calculations
- Comprehensive analysis of different cell types and patterns
- Visualization of analysis results

Usage:

# 基本使用
from spatial_analyzer import SpatialPatternAnalyzer, CellData, PatternType

# 初始化分析器
analyzer = SpatialPatternAnalyzer()

# 準備資料
data = analyzer.preprocess_data(x_coords, y_coords, cell_types)

# 執行分析
results = analyzer.analyze_cell_types(data, pattern_types=[PatternType.ORIGINAL, PatternType.RANDOM])

# 視覺化結果
analyzer.plot_results(results)



"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.spatial import cKDTree
import math
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
from convex_hull_pattern_generator import PatternGenerator, PatternType as CHPatternType

class PatternType(Enum):
    """Available spatial pattern types for simulation"""
    ORIGINAL = "original"
    RANDOM = "random" 
    CLUSTER = "cluster"
    REGULAR = "regular"
    
    @classmethod
    def from_ch_pattern_type(cls, ch_type: CHPatternType) -> 'PatternType':
        """Convert from convex_hull_pattern_generator PatternType"""
        return cls(ch_type.value)

class SampleShape(Enum):
    """Available sample shapes for boundary correction"""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"

@dataclass
class RipleyResult:
    """Container for Ripley analysis results"""
    k_values: np.ndarray
    l_values: np.ndarray
    radii: np.ndarray
    points_analyzed: int
    sample_size: float

@dataclass
class CellData:
    """Container for cell coordinate and type data"""
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None
    cell_types: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate data consistency"""
        if len(self.x) != len(self.y):
            raise ValueError("X and Y coordinates must have same length")
        if self.z is not None and len(self.z) != len(self.x):
            raise ValueError("Z coordinates must have same length as X and Y")
        if self.cell_types is not None and len(self.cell_types) != len(self.x):
            raise ValueError("Cell types must have same length as coordinates")

class SpatialPatternAnalyzer:
    """
    Main class for spatial point pattern analysis using Ripley's K and L functions
    """
    
    def __init__(self, scan_scope: float = 6.0, shrink_factor: float = 0.8, 
                 n_radii: int = 100):
        """
        Initialize the analyzer with configuration parameters
        
        Args:
            scan_scope: Scanning radius as multiple of median distance
            shrink_factor: Factor to reduce sample size to avoid edge effects
            n_radii: Number of radii points for analysis
        """
        self.scan_scope = scan_scope
        self.shrink_factor = shrink_factor
        self.n_radii = n_radii
        self.pattern_generator = PatternGenerator()
        
    def preprocess_data(self, x: np.ndarray, y: np.ndarray, 
                       cell_types: Optional[np.ndarray] = None) -> CellData:
        """
        Clean and preprocess cell coordinate data
        
        Args:
            x, y: Cell coordinates
            cell_types: Optional cell type labels
            
        Returns:
            CellData object with cleaned data
        """
        # Convert to numpy arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if cell_types is not None:
            cell_types = np.asarray(cell_types)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        if cell_types is not None:
            cell_types = cell_types[valid_mask]
        
        # Remove duplicate coordinates
        coords = np.column_stack((x, y))
        unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)
        
        if len(unique_coords) < len(coords):
            warnings.warn(f"Removed {len(coords) - len(unique_coords)} duplicate coordinates")
            x = x[unique_indices]
            y = y[unique_indices]
            if cell_types is not None:
                cell_types = cell_types[unique_indices]
        
        return CellData(x=x, y=y, cell_types=cell_types)
    
    def calculate_distances(self, data: CellData) -> Tuple[float, float]:
        """
        Calculate median and minimum nearest neighbor distances
        
        Args:
            data: CellData object
            
        Returns:
            Tuple of (median_distance, min_distance)
        """
        coords = np.column_stack((data.x, data.y))
        tree = cKDTree(coords)
        
        # Query 2 nearest neighbors (including self)
        distances, _ = tree.query(coords, k=2)
        nearest_distances = distances[:, 1]  # Exclude self (distance 0)
        
        # Remove any zero distances (duplicates that weren't caught)
        nearest_distances = nearest_distances[nearest_distances > 0]
        
        if len(nearest_distances) == 0:
            raise ValueError("No valid nearest neighbor distances found")
            
        return np.median(nearest_distances), np.min(nearest_distances)
    
    def make_tree(self, x: np.ndarray, y: np.ndarray, 
                  z: Optional[np.ndarray] = None) -> Tuple[cKDTree, int]:
        """
        Create spatial tree for efficient neighbor queries
        
        Args:
            x, y: 2D coordinates (required)
            z: 3D coordinate (optional)
            
        Returns:
            Tuple of (tree, dimensions)
        """
        if z is not None:
            points = np.column_stack((x, y, z))
            dimensions = 3
        else:
            points = np.column_stack((x, y))
            dimensions = 2
            
        return cKDTree(points), dimensions
    
    def calculate_overlap(self, point: np.ndarray, bounding_size: float, 
                         radius: float, dimensions: int, 
                         sample_shape: SampleShape = SampleShape.CIRCLE) -> float:
        """
        Calculate overlap volume for boundary correction
        
        Args:
            point: Point coordinates
            bounding_size: Size of bounding region
            radius: Scoring radius
            dimensions: Number of spatial dimensions
            sample_shape: Shape of sampling region
            
        Returns:
            Overlap volume
        """
        if dimensions == 2 and sample_shape == SampleShape.CIRCLE:
            d = np.linalg.norm(point)
            
            # Point outside boundary
            if d > bounding_size:
                return 0.0
            
            # Scoring circle entirely within boundary
            if d <= abs(radius - bounding_size):
                return np.pi * radius**2
            
            # Partial overlap calculation
            r, R = radius, bounding_size
            alpha = np.arccos((d**2 + r**2 - R**2) / (2*d*r))
            beta = np.arccos((d**2 + R**2 - r**2) / (2*d*R))
            
            overlap = (r**2 * alpha + R**2 * beta - 
                      0.5 * (r**2 * np.sin(2*alpha) + R**2 * np.sin(2*beta)))
            
            return max(0, overlap)
        
        # Add other dimension/shape combinations as needed
        raise NotImplementedError(f"Overlap calculation for {dimensions}D {sample_shape.value} not implemented")
    
    def calculate_ripley(self, data: CellData, radii: Optional[np.ndarray] = None,
                        sample_size: Optional[float] = None, auto_center: bool = True,
                        boundary_correct: bool = True, 
                        sample_shape: SampleShape = SampleShape.CIRCLE) -> RipleyResult:
        """
        Calculate Ripley's K and L functions
        
        Args:
            data: CellData object with coordinates
            radii: Array of radii to analyze (auto-generated if None)
            sample_size: Size of sampling region (auto-calculated if None)
            auto_center: Whether to center coordinates at origin
            boundary_correct: Whether to apply boundary correction
            sample_shape: Shape of sampling region
            
        Returns:
            RipleyResult object with analysis results
        """
        x, y = data.x.copy(), data.y.copy()
        
        # Auto-center coordinates
        if auto_center:
            x = x - np.mean(x)
            y = y - np.mean(y)
        
        # Auto-calculate sample size
        if sample_size is None:
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            sample_size = min(x_range, y_range) * self.shrink_factor
        
        # Auto-generate radii
        if radii is None:
            radii = np.linspace(0, sample_size, self.n_radii)[1:]  # Exclude 0
        
        # Create spatial tree
        tree, dimensions = self.make_tree(x, y)
        
        # Calculate points within sample boundary
        if boundary_correct and sample_shape == SampleShape.CIRCLE:
            center_point = [0, 0] if dimensions == 2 else [0, 0, 0]
            boundary_indices = tree.query_ball_point(center_point, sample_size)
            points_analyzed = len(boundary_indices)
        else:
            points_analyzed = len(x)
        
        k_values = []
        
        for radius in radii:
            score_volume = np.pi * radius**2  # Circle area
            boundary_volume = np.pi * sample_size**2  # Sample area
            
            total_count = 0
            
            for i, (xi, yi) in enumerate(zip(x, y)):
                if boundary_correct:
                    overlap = self.calculate_overlap([xi, yi], sample_size, radius, 
                                                   dimensions, sample_shape)
                    if overlap > 0:
                        boundary_correction = overlap / score_volume
                        
                        # Count neighbors within radius
                        neighbor_indices = tree.query_ball_point([xi, yi], radius)
                        
                        # Filter neighbors within sample boundary
                        if sample_shape == SampleShape.CIRCLE:
                            valid_neighbors = [idx for idx in neighbor_indices 
                                             if idx != i and np.linalg.norm([x[idx], y[idx]]) <= sample_size]
                        else:
                            valid_neighbors = [idx for idx in neighbor_indices if idx != i]
                        
                        total_count += len(valid_neighbors) / boundary_correction
                else:
                    neighbor_indices = tree.query_ball_point([xi, yi], radius)
                    total_count += len(neighbor_indices) - 1  # Exclude self
            
            k_value = (boundary_volume * total_count) / (len(x) ** 2)
            k_values.append(k_value)
        
        k_values = np.array(k_values)
        l_values = np.sqrt(k_values / np.pi)
        
        return RipleyResult(k_values=k_values, l_values=l_values, 
                          radii=radii, points_analyzed=points_analyzed,
                          sample_size=sample_size)
    
    def generate_pattern(self, data: CellData, n_cells: int, 
                        pattern_type: PatternType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate artificial spatial patterns using convex_hull_pattern_generator
        
        Args:
            data: Original CellData for reference
            n_cells: Number of cells to generate
            pattern_type: Type of pattern to generate
            
        Returns:
            Tuple of (x_coords, y_coords) for generated pattern
        """
        # 轉換 PatternType 為 convex_hull_pattern_generator 的 PatternType
        ch_pattern_type = CHPatternType(pattern_type.value)
        
        # 使用 convex_hull_pattern_generator 的 PatternGenerator
        return self.pattern_generator.generate_pattern(data, n_cells, ch_pattern_type)
    
    def analyze_cell_types(self, data: CellData, 
                          pattern_types: List[PatternType] = None) -> Dict:
        """
        Comprehensive analysis of different cell types and patterns
        
        Args:
            data: CellData with cell type information
            pattern_types: List of patterns to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if data.cell_types is None:
            raise ValueError("Cell type information required for this analysis")
        
        if pattern_types is None:
            pattern_types = [PatternType.ORIGINAL]
        
        results = {}
        unique_types, type_counts = np.unique(data.cell_types, return_counts=True)
        
        # Calculate reference distances
        median_dist, min_dist = self.calculate_distances(data)
        
        for pattern_type in pattern_types:
            pattern_results = {}
            
            for cell_type, count in zip(unique_types, type_counts):
                # Extract cells of this type
                type_mask = data.cell_types == cell_type
                type_data = CellData(
                    x=data.x[type_mask], 
                    y=data.y[type_mask],
                    cell_types=data.cell_types[type_mask]
                )
                # 如果只有一種cell_types，data與type_data相等

                # Generate pattern if not original
                if pattern_type != PatternType.ORIGINAL:
                    x_pattern, y_pattern = self.generate_pattern(data, count, pattern_type)
                    type_data = CellData(
                        x=x_pattern, 
                        y=y_pattern,
                        cell_types=np.array([cell_type] * count)  # 添加 cell_types
                    )

                
                # Calculate Ripley functions
                result = self.calculate_ripley(type_data)
                
                # Calculate local vs Global L function differences
                local_analysis = self._calculate_local_l_differences(
                    type_data, data, median_dist, static_global_L=(result if len(unique_types)==1 else None)
                )
                
                pattern_results[cell_type] = {
                    'ripley_result': result,
                    'local_analysis': local_analysis,
                    'cell_count': count
                }
            
            results[pattern_type.value] = pattern_results
        
        return results
    
    def _calculate_local_l_differences(self, type_data: CellData, 
                                     all_data: CellData, median_dist: float, static_global_L:RipleyResult = None) -> Dict:
        """
        Calculate local L() vs global L() differences in each cell type
        """
        scan_radii = np.arange(median_dist, median_dist * (1 + self.scan_scope), 
                              median_dist / 3)
        
        local_values = []
        
        for i in tqdm(range(len(type_data.x)), desc="Local analysis"):
            # Center coordinates on current cell
            centered_type_x = type_data.x - type_data.x[i]
            centered_type_y = type_data.y - type_data.y[i]
            
            # all_data
            centered_all_x = all_data.x - type_data.x[i]
            centered_all_y = all_data.y - type_data.y[i]
            
            # Filter points within scan scope
            scan_radius = median_dist * (self.scan_scope + 1)
            type_mask = ((centered_type_x**2 + centered_type_y**2) <= scan_radius**2)
            all_mask = ((centered_all_x**2 + centered_all_y**2) <= scan_radius**2)
            
            if np.sum(type_mask) > 2 and np.sum(all_mask) > 2:
                # Calculate L functions for type-specific and all cells
                type_centered = CellData(
                    x=centered_type_x[type_mask], 
                    y=centered_type_y[type_mask]
                )
                
                type_result = self.calculate_ripley(
                    type_centered, radii=scan_radii, 
                    sample_size=median_dist * self.scan_scope,
                    auto_center=False
                )
                
                if static_global_L is not None:
                    # 使用static_global_L替代all_data的計算
                    l_diff = np.mean(type_result.l_values)-np.mean(static_global_L.l_values)

                else:
                    all_centered = CellData(
                        x=centered_all_x[all_mask], 
                        y=centered_all_y[all_mask]
                    )
                    all_result = self.calculate_ripley(
                        all_centered, radii=scan_radii,
                        sample_size=median_dist * self.scan_scope, 
                        auto_center=False
                    )

                    # Calculate mean difference
                    l_diff = np.mean(type_result.l_values - all_result.l_values)

                local_values.append([type_data.x[i], type_data.y[i], l_diff])
            else:
                local_values.append([type_data.x[i], type_data.y[i], 0])
        
        return {
            'coordinates': np.array(local_values),
            'scan_radii': scan_radii,
            'median_distance': median_dist
        }
    
    def plot_results(self, results: Dict, title: str = "Spatial Analysis Results"):
        """
        Create comprehensive visualization of analysis results
        
        Args:
            results: Results dictionary from analyze_cell_types
            title: Plot title
        """
        n_patterns = len(results)
        
        # 計算最佳的長寬比
        total_cell_types = sum(len(pattern_data) for pattern_data in results.values())
        target_aspect_ratio = 3/2  # 目標長寬比
        
        # 計算所需的列數和行數
        n_cols = n_patterns + 1  # 額外一列用於比較
        n_rows = 2  # 固定兩行：散點圖和L函數圖
        
        # 計算每個子圖的寬度和高度
        fig_width = 8 * n_cols
        fig_height = 8 * n_rows
        
        # 創建主圖形
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.set_dpi(300)
        plt.subplots_adjust(
            right=0.85,  # 為 colorbar 留出空間
            wspace=0.3,  # 增加水平間距
            hspace=0.4   # 增加垂直間距
        )
        
        # 創建上下兩個 subfigures
        subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])
        
        # 先計算第一張圖的座標範圍
        first_pattern = next(iter(results.values()))
        first_cell_type = next(iter(first_pattern.values()))
        first_coords = first_cell_type['local_analysis']['coordinates']
        x_min = np.min(first_coords[:, 0])
        x_max = np.max(first_coords[:, 0])
        y_min = np.min(first_coords[:, 1])
        y_max = np.max(first_coords[:, 1])
        
        # 計算第一張圖的長寬比
        first_aspect_ratio = (x_max - x_min) / (y_max - y_min)
        
        # 根據長寬比和模式數量決定 subplots 的排列
        if first_aspect_ratio < 1.5:  # 如果圖形較寬
            scatter_rows = 1
            scatter_cols = n_patterns
        else:  # 如果圖形較高
            scatter_rows = min(2, n_patterns)
            scatter_cols = (n_patterns + scatter_rows - 1) // scatter_rows
        
        # 在上方 subfigure 中創建散點圖
        scatter_axes = subfigs[0].subplots(scatter_rows, scatter_cols)
        if n_patterns == 1:
            scatter_axes = np.array([scatter_axes])
        scatter_axes = scatter_axes.flatten()
        
        # 在下方 subfigure 中創建 L 函數圖
        lfunc_axes = subfigs[1].subplots(1, n_patterns + 1)
        if n_patterns == 1:
            lfunc_axes = np.array([lfunc_axes[0]])
        
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # 用於存儲所有模式的 L 函數數據
        all_patterns_data = {}
        
        # 用於存儲所有散點圖的範圍
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        c_min, c_max = float('inf'), float('-inf')  # 用於存儲colorbar的範圍
        
        # 設定給不同cell_type用的標記
        scatter_mark=['o','s','D','*']
        scatter_edgecolors=[None,'black','red','green','blue']
        
        # 第一次遍歷：收集所有數據的範圍
        for pattern_name, pattern_data in results.items():
            for cell_type, type_data in pattern_data.items():
                local_coords = type_data['local_analysis']['coordinates']
                
                # 更新座標範圍
                x_min = min(x_min, np.min(local_coords[:, 0]))
                x_max = max(x_max, np.max(local_coords[:, 0]))
                y_min = min(y_min, np.min(local_coords[:, 1]))
                y_max = max(y_max, np.max(local_coords[:, 1]))
                c_min = min(c_min, np.min(local_coords[:, 2]))
                c_max = max(c_max, np.max(local_coords[:, 2]))
        
        # 第二次遍歷：繪製圖形
        for col, (pattern_name, pattern_data) in enumerate(results.items()):
            ax_map = scatter_axes[col]
            ax_func = lfunc_axes[col]
            
            # Plot spatial distribution
            for i, (cell_type, type_data) in enumerate(pattern_data.items()):
                local_coords = type_data['local_analysis']['coordinates']
                
                # set marker type and color
                sm=scatter_mark[i%len(scatter_mark)]
                smc=scatter_edgecolors[(i//len(scatter_mark))%len(scatter_edgecolors)]
                
                # Scatter plot colored by L function difference
                scatter = ax_map.scatter(
                    local_coords[:, 0], local_coords[:, 1],
                    c=local_coords[:, 2], cmap='RdYlGn', marker=sm, edgecolors=smc,
                    s=20, alpha=0.7, label=f"{cell_type}-{pattern_name.title()} (N={len(local_coords)})",
                    vmin=c_min, vmax=c_max  # 設定統一的顏色範圍
                )
                
                # 存儲 L 函數數據用於後續比較
                if cell_type not in all_patterns_data:
                    all_patterns_data[cell_type] = {}
                all_patterns_data[cell_type][pattern_name] = {
                    'radii': type_data['ripley_result'].radii,
                    'l_values': type_data['ripley_result'].l_values
                }
                
                # Plot L function
                ripley_result = type_data['ripley_result']
                ax_func.plot(
                    ripley_result.radii, ripley_result.l_values,
                    color=colors[i], label=f"{cell_type}", linewidth=2
                )
            
            # Add theoretical CSR line
            ax_func.plot(
                ripley_result.radii, ripley_result.radii, 
                'k--', alpha=0.5, label='Theor. random'
            )
            
            ax_map.set_title(f'{pattern_name.title()} Pattern')
            ax_map.set_xlabel('X coordinate')
            ax_map.set_ylabel('Y coordinate')
            # 將圖例放在散點圖下方
            ax_map.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            ax_map.set_aspect('equal')
            
            ax_func.set_title(f'L Function - {pattern_name.title()}')
            ax_func.set_xlabel('Radius')
            ax_func.set_ylabel('L(r)')
            # 將 L 函數圖的圖例放在左上角
            ax_func.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            ax_func.grid(True, alpha=0.3)
        
        # 統一所有散點圖的座標範圍，並反轉 Y 軸
        for col in range(n_patterns):
            ax_map = scatter_axes[col]
            ax_map.set_xlim(x_min, x_max)
            ax_map.set_ylim(y_max, y_min)  # 反轉 Y 軸
        
        # 添加共用的 colorbar，放在所有散點圖的右側
        cbar_ax = fig.add_axes([0.88, 0.55, 0.02, 0.3])  # 調整位置到散點圖區域
        cb = fig.colorbar(scatter, cax=cbar_ax)
        cb.set_label('local L() - global L()') # local L() of a cell type vs global L()
        
        # 在最後一列添加比較所有模式的 L 函數圖
        if n_patterns > 1:
            ax_compare = lfunc_axes[-1]
            for cell_type, pattern_data in all_patterns_data.items():
                for pattern_name, data in pattern_data.items():
                    ax_compare.plot(
                        data['radii'], data['l_values'],
                        label=f"{cell_type}-{pattern_name.title()}",
                        linewidth=2
                    )
            
            # 添加理論線
            ax_compare.plot(
                data['radii'], data['radii'],
                'k--', alpha=0.5, label='Theor. random'
            )
            
            ax_compare.set_title('L Function Comparison')
            ax_compare.set_xlabel('Radius')
            ax_compare.set_ylabel('L(r)')
            # 將比較圖的圖例也放在左上角
            ax_compare.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            ax_compare.grid(True, alpha=0.3)
        
        plt.suptitle(title, y=0.98, fontsize=16)
        
        return fig

# Example usage and testing functions
def example_usage():
    """
    Example of how to use the improved SpatialPatternAnalyzer
    """
    # Generate sample data
    #np.random.seed(42)
    n_cells = 200
    
    # Create clustered pattern
    centers = [(10, 10),]
    x_coords, y_coords, cell_types = [], [], []
    
    for i, (cx, cy) in enumerate(centers):
        n_cluster = n_cells // 3
        cluster_x = np.random.normal(cx, 5, n_cluster)
        cluster_y = np.random.normal(cy, 5, n_cluster)
        cluster_types = [f'Type_{i+1}'] * n_cluster
        
        x_coords.extend(cluster_x)
        y_coords.extend(cluster_y)
        cell_types.extend(cluster_types)
    
    # Create analyzer and run analysis
    analyzer = SpatialPatternAnalyzer()
    
    # Preprocess data
    data = analyzer.preprocess_data(
        np.array(x_coords), np.array(y_coords), np.array(cell_types)
    )
    
    # Run comprehensive analysis
    results = analyzer.analyze_cell_types(
        data, 
        pattern_types=[PatternType.ORIGINAL,PatternType.RANDOM,PatternType.CLUSTER]
    )
    
    # Create visualization
    fig = analyzer.plot_results(results, "Example Spatial Analysis")
    
    return analyzer, results, fig

if __name__ == "__main__":
    # Run example
    analyzer, results, fig = example_usage()
    #fig.savefig("spatial_analysis_results.png")
    plt.show()
    print("Example Analysis completed successfully!")
