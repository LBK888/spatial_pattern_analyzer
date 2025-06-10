"""
根據原始座標數據點產生三種不同對比pattern -- v0.9 2025-06-09
分離評估函數：
evaluate_spatial_similarity: 使用 Wasserstein 距離評估
evaluate_median_similarity: 使用中位數距離比率評估
Wasserstein 距離（W-Dist）越小表示越相似，而中位數相似度（M-Sim）越接近 1 表示越相似。這兩種評估方式可以互補，提供不同角度的相似度評估。
"""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist
from matplotlib.path import Path
from typing import Tuple, List, Optional
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import wasserstein_distance

class PatternType(Enum):
    ORIGINAL = "original"
    RANDOM = "random"
    CLUSTER = "cluster"
    REGULAR = "regular"

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

@dataclass
class PatternParameters:
    """模式生成參數配置"""
    density_factor: float = 1.15  # 蜂巢格密度因子
    variance_min: float = 0.05    # 最小變異度
    variance_max: float = 0.1     # 最大變異度
    cluster_radius_factor: float = 0.25  # 聚集半徑因子
    min_clusters: int = 6         # 最小聚集數
    max_clusters: int = 36        # 最大聚集數
    density_tolerance: float = 0.1  # 密度容差
    dbscan_eps: float = 0.8       # DBSCAN 聚類參數，要尋找的半徑
    dbscan_min_samples: int = 5   # DBSCAN 最小樣本數
    densify_k: int = 5             # 要增加的凸包多邊形點

def compute_minimum_bounding_polygon(points: np.ndarray, densify_k: int = 0, original_points: np.ndarray = None) -> np.ndarray:
    """
    計算一組2D點的最小包圍凸多邊形，可選擇是否細緻化邊緣
    
    Args:
        points: 形狀為 (n, 2) 的numpy陣列，包含 x, y 座標
        densify_k: 每條邊上要插入的內部點數（不含端點），0表示不細緻化
        original_points: 原始數據點，用於調整新增點的位置
        
    Returns:
        形狀為 (m, 2) 的numpy陣列，包含凸包多邊形的頂點座標
    """
    if len(points) < 3:
        raise ValueError("至少需要3個點才能形成多邊形")
    
    # 使用 ConvexHull 計算凸包
    hull = ConvexHull(points)
    
    # 獲取凸包頂點
    hull_points = points[hull.vertices]
    
    # 如果需要細緻化
    if densify_k > 0:
        return densify_minimum_bounding_polygon(hull_points, densify_k, original_points)
    
    return hull_points


def densify_minimum_bounding_polygon(points: np.ndarray, k: int = 30, original_points: np.ndarray = None) -> np.ndarray:
    """
    在凸包邊緣上增加點，並根據原始數據調整位置使其更貼近數據分布
    
    Args:
        points: M×2 陣列，按凸包順序排列，且 points[0] != points[-1]
        k: 每條邊上要插入的內部點數（不含端點）
        original_points: 原始數據點，用於調整新增點的位置
        
    Returns:
        增加點後的凸包頂點陣列，包含原始頂點和調整後的新點
    """
    dense = []
    M = len(points)
    
    # 如果沒有提供原始數據點，則使用簡單的線性插值
    if original_points is None:
        for i in range(M):
            p0 = points[i]
            p1 = points[(i+1) % M]
            # 先加入原始頂點
            dense.append(p0)
            # 再加入插值點
            ts = np.linspace(0, 1, k+2)[1:-1]  # 去掉0和1，避免重複端點
            for t in ts:
                dense.append((1-t)*p0 + t*p1)
        return np.vstack(dense)
    
    # 使用原始數據點來調整新增點的位置
    for i in range(M):
        p0 = points[i]
        p1 = points[(i+1) % M]
        
        # 先加入原始頂點
        dense.append(p0)
        
        # 計算當前邊的方向向量和長度
        edge_vector = p1 - p0
        edge_length = np.linalg.norm(edge_vector)
        edge_direction = edge_vector / edge_length
        
        # 向量化計算所有原始點到當前邊的投影和距離
        v = original_points - p0
        proj = np.dot(v, edge_direction)
        perp = v - np.outer(proj, edge_direction)
        dist = np.linalg.norm(perp, axis=1)
        
        # 找出在邊附近的點
        mask = (proj >= 0) & (proj <= edge_length) & (dist < edge_length * 0.5)
        nearby_indices = np.where(mask)[0]
        
        if len(nearby_indices) > 0:
            nearby_points = original_points[nearby_indices]
            nearby_proj = proj[nearby_indices] / edge_length
            nearby_dist = dist[nearby_indices]
            
            # 生成新點的位置參數
            ts = np.linspace(0, 1, k+2)[1:-1]
            
            for t in ts:
                # 基本位置（線性插值）
                base_point = (1-t)*p0 + t*p1
                
                # 找出時間參數附近的點
                time_mask = np.abs(nearby_proj - t) < 0.2
                if np.any(time_mask):
                    # 找出最近的點
                    valid_dist = nearby_dist[time_mask]
                    valid_points = nearby_points[time_mask]
                    closest_idx = np.argmin(valid_dist)
                    closest_point = valid_points[closest_idx]
                    
                    # 檢查這個點是否已經在列表中
                    is_duplicate = False
                    for existing_point in dense:
                        if np.allclose(existing_point, closest_point, atol=1e-6):
                            is_duplicate = True
                            break
                    
                    # 如果不是重複點，則加入
                    if not is_duplicate:
                        # 計算調整向量，如有需要可以增加這步運算
                        '''
                        adjustment = (valid_points[closest_idx] - base_point) * 0.3  #只向內收縮30%
                        base_point += adjustment
                        dense.append(base_point)
                        '''

                        dense.append(closest_point)
    
    return np.vstack(dense)

def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    檢查點是否在多邊形內部
    
    Args:
        point: (x, y) 座標
        polygon: 多邊形頂點座標陣列
        
    Returns:
        True 如果點在多邊形內部
    """
    path = Path(polygon)
    return path.contains_point(point)

def generate_random_points_in_polygon(polygon: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    在多邊形內隨機生成指定數量的點
    
    Args:
        polygon: 多邊形頂點座標
        n_points: 要生成的點數量
        
    Returns:
        (x_coords, y_coords) 元組
    """
    # 獲取多邊形的邊界框
    x_min, x_max = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    y_min, y_max = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    
    x_coords = []
    y_coords = []
    
    # 使用拒絕採樣法
    while len(x_coords) < n_points:
        # 在邊界框內隨機生成點
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        # 檢查點是否在多邊形內
        if point_in_polygon((x, y), polygon):
            x_coords.append(x)
            y_coords.append(y)
    
    return np.array(x_coords), np.array(y_coords)

def calculate_spatial_variance(points: np.ndarray) -> float:
    """
    計算原始數據點的空間變異度
    
    Args:
        points: 形狀為 (n, 2) 的點座標陣列
        
    Returns:
        空間變異度（用於控制蜂巢格的不規則程度）
    """
    if len(points) < 2:
        return 0.1
    
    # 計算最近鄰距離
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(6, len(points)))
    
    # 使用第2到第6個最近鄰的距離（排除自己）
    neighbor_distances = distances[:, 1:].flatten()
    neighbor_distances = neighbor_distances[neighbor_distances > 0]
    
    if len(neighbor_distances) == 0:
        return 0.1
    
    # 計算變異係數（標準差/平均值）
    mean_dist = np.mean(neighbor_distances)
    std_dist = np.std(neighbor_distances)
    
    if mean_dist == 0:
        return 0.1
    
    variance_coefficient = std_dist / mean_dist
    
    # 將變異係數限制在合理範圍內 (0.05 - 0.4)
    return np.clip(variance_coefficient, 0.05, 0.4)

def generate_regular_points_in_polygon(polygon: np.ndarray, n_points: int, 
                                     original_points: np.ndarray,
                                     params: PatternParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    在多邊形內生成交錯蜂巢格模式的點（帶有自然變異）
    
    Args:
        polygon: 多邊形頂點座標
        n_points: 要生成的點數量
        original_points: 原始數據點，用於計算變異度
        params: 模式生成參數
        
    Returns:
        (x_coords, y_coords) 元組
    """
    # 獲取多邊形的邊界框
    x_min, x_max = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    y_min, y_max = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    
    # 計算空間變異度
    variance = calculate_spatial_variance(original_points)
    variance = np.clip(variance, params.variance_min, params.variance_max)
    
    # 計算原始密度
    orig_density = calculate_density(original_points, polygon)
    
    # 估算蜂巢格間距
    area_bbox = (x_max - x_min) * (y_max - y_min)
    base_spacing = np.sqrt(area_bbox / (n_points * params.density_factor))
    
    # 蜂巢格的行間距（y方向）
    row_spacing = base_spacing * np.sqrt(3) / 2
    
    # 生成交錯蜂巢格點
    grid_points = []
    y_current = y_min
    row_index = 0
    
    while y_current <= y_max:
        # 決定這一行的x起始位置（交錯排列）
        if row_index % 2 == 0:
            x_start = x_min
        else:
            x_start = x_min + base_spacing / 2
        
        # 在這一行生成點
        x_current = x_start
        while x_current <= x_max:
            # 添加隨機變異
            noise_x = np.random.normal(0, base_spacing * variance)
            noise_y = np.random.normal(0, base_spacing * variance)
            
            x_perturbed = x_current + noise_x
            y_perturbed = y_current + noise_y
            
            # 檢查擾動後的點是否在多邊形內
            if point_in_polygon((x_perturbed, y_perturbed), polygon):
                grid_points.append((x_perturbed, y_perturbed))
            
            x_current += base_spacing
        
        y_current += row_spacing
        row_index += 1
    
    # 如果生成的點數不夠，動態調整間距
    adjustment_factor = 0.9
    attempts = 0
    max_attempts = 5
    
    while len(grid_points) < n_points * 0.8 and attempts < max_attempts:
        attempts += 1
        base_spacing *= adjustment_factor
        row_spacing = base_spacing * np.sqrt(3) / 2
        
        grid_points = []
        y_current = y_min
        row_index = 0
        
        while y_current <= y_max:
            if row_index % 2 == 0:
                x_start = x_min
            else:
                x_start = x_min + base_spacing / 2
            
            x_current = x_start
            while x_current <= x_max:
                noise_x = np.random.normal(0, base_spacing * variance)
                noise_y = np.random.normal(0, base_spacing * variance)
                
                x_perturbed = x_current + noise_x
                y_perturbed = y_current + noise_y
                
                if point_in_polygon((x_perturbed, y_perturbed), polygon):
                    grid_points.append((x_perturbed, y_perturbed))
                
                x_current += base_spacing
            
            y_current += row_spacing
            row_index += 1
    
    # 選擇最接近所需數量的點
    if len(grid_points) >= n_points:
        # 使用空間分佈選擇點，避免過度聚集
        selected_points = []
        remaining_points = grid_points.copy()
        
        # 先隨機選擇一個起始點
        if remaining_points:
            start_idx = np.random.randint(len(remaining_points))
            selected_points.append(remaining_points.pop(start_idx))
        
        # 使用最遠點採樣策略選擇剩餘點
        while len(selected_points) < n_points and remaining_points:
            if len(selected_points) < n_points // 3:
                # 前1/3使用最遠點採樣
                distances_to_selected = []
                for candidate in remaining_points:
                    min_dist = min([np.linalg.norm(np.array(candidate) - np.array(selected)) 
                                  for selected in selected_points])
                    distances_to_selected.append(min_dist)
                
                # 選擇距離已選點最遠的點
                max_dist_idx = np.argmax(distances_to_selected)
                selected_points.append(remaining_points.pop(max_dist_idx))
            else:
                # 後2/3隨機選擇
                random_idx = np.random.randint(len(remaining_points))
                selected_points.append(remaining_points.pop(random_idx))
    else:
        # 如果蜂巢格點不夠，用隨機點補充
        selected_points = grid_points.copy()
        remaining = n_points - len(selected_points)
        if remaining > 0:
            x_random, y_random = generate_random_points_in_polygon(polygon, remaining)
            for i in range(remaining):
                selected_points.append((x_random[i], y_random[i]))
    
    x_coords = [p[0] for p in selected_points]
    y_coords = [p[1] for p in selected_points]
    
    return np.array(x_coords), np.array(y_coords)

def generate_cluster_points_in_polygon(polygon: np.ndarray, n_points: int,
                                     original_points: np.ndarray,
                                     params: PatternParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    在多邊形內生成聚集模式的點
    
    Args:
        polygon: 多邊形頂點座標
        n_points: 要生成的點數量
        original_points: 原始數據點，用於分析聚集特徵
        params: 模式生成參數
        
    Returns:
        (x_coords, y_coords) 元組
    """
    # 分析原始數據的聚集特徵
    n_clusters = analyze_clusters(original_points, params)
    #print("DBSCAN n=",n_clusters)
    n_clusters = max(params.min_clusters, 
                    min(params.max_clusters, n_clusters))
    
    # 在多邊形內隨機選擇聚集中心
    cluster_centers_x, cluster_centers_y = generate_random_points_in_polygon(polygon, n_clusters)
    
    # 計算多邊形的特徵尺寸（用於設定聚集半徑）
    x_range = np.max(polygon[:, 0]) - np.min(polygon[:, 0])
    y_range = np.max(polygon[:, 1]) - np.min(polygon[:, 1])
    avg_range = (x_range + y_range) / 2
    cluster_radius = avg_range / (n_clusters ** 0.5) * params.cluster_radius_factor
    #print("cluster_radius=",cluster_radius)
    
    # 為每個聚集分配點數
    points_per_cluster = n_points // n_clusters
    remaining_points = n_points % n_clusters
    
    # 一次性生成所有聚集的點
    total_points = n_points
    angles = np.random.uniform(0, 2 * np.pi, total_points)
    radii = np.random.exponential(cluster_radius / 3, total_points)
    
    # 計算每個點屬於哪個聚集
    cluster_indices = np.repeat(np.arange(n_clusters), points_per_cluster)
    if remaining_points > 0:
        cluster_indices = np.append(cluster_indices, np.arange(remaining_points))
    
    # 計算每個點的座標
    x = cluster_centers_x[cluster_indices] + radii * np.cos(angles)
    y = cluster_centers_y[cluster_indices] + radii * np.sin(angles)
    
    # 使用向量化操作檢查點是否在多邊形內
    points = np.column_stack((x, y))
    path = Path(polygon)
    mask = path.contains_points(points)
    
    # 過濾出在多邊形內的點
    valid_points = points[mask]
    
    # 如果點數不夠，在多邊形內隨機補充
    if len(valid_points) < n_points:
        remaining = n_points - len(valid_points)
        x_random, y_random = generate_random_points_in_polygon(polygon, remaining)
        valid_points = np.vstack((valid_points, np.column_stack((x_random, y_random))))
    
    return valid_points[:, 0], valid_points[:, 1]

def calculate_polygon_area(polygon: np.ndarray) -> float:
    """計算多邊形面積"""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_density(points: np.ndarray, polygon: np.ndarray) -> float:
    """計算點密度"""
    area = calculate_polygon_area(polygon)
    return len(points) / area

def analyze_clusters(points: np.ndarray, params: PatternParameters) -> int:
    """分析原始數據的聚集特徵
    params.dbscan_eps
    """
    
    clustering = DBSCAN(eps=calculate_median_distance(points), 
                       min_samples=params.dbscan_min_samples).fit(points)
    return len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

def validate_density(original_points: np.ndarray, 
                    generated_points: np.ndarray, 
                    polygon: np.ndarray,
                    params: PatternParameters) -> bool:
    """驗證生成點的密度是否與原始數據相近"""
    orig_density = calculate_density(original_points, polygon)
    gen_density = calculate_density(generated_points, polygon)
    return abs(orig_density - gen_density) / orig_density < params.density_tolerance

def calculate_median_distance(points: np.ndarray) -> float:
    """
    計算點集合中點與點之間的中位數距離
    
    Args:
        points: 形狀為 (n, 2) 的點座標陣列
        
    Returns:
        點間距的中位數
    """
    # 計算所有點對之間的距離
    distances = pdist(points)
    # 返回中位數距離
    return np.median(distances)

def evaluate_spatial_similarity(original_points: np.ndarray, 
                              generated_points: np.ndarray) -> float:
    """
    使用 Wasserstein 距離評估生成點與原始點的空間相似度
    
    Args:
        original_points: 原始數據點
        generated_points: 生成的點
        
    Returns:
        Wasserstein 距離值（越小表示越相似）
    """
    return wasserstein_distance(original_points.flatten(), 
                              generated_points.flatten())

def evaluate_median_similarity(original_points: np.ndarray, 
                             generated_points: np.ndarray) -> float:
    """
    使用中位數距離評估生成點與原始點的空間相似度
    
    Args:
        original_points: 原始數據點
        generated_points: 生成的點
        
    Returns:
        相似度分數 (0-1之間，1表示最相似)
    """
    # 計算原始點和生成點的點間距中位數
    orig_median_dist = calculate_median_distance(original_points)
    gen_median_dist = calculate_median_distance(generated_points)
    
    # 計算中位數距離的差異比例
    dist_ratio = min(orig_median_dist, gen_median_dist) / max(orig_median_dist, gen_median_dist)
    
    return dist_ratio

class PatternGenerator:
    """修改後的模式生成器類別"""
    
    def __init__(self, params: Optional[PatternParameters] = None):
        """初始化模式生成器
        
        Args:
            params: 模式生成參數配置，如果為 None 則使用預設值
        """
        self.params = params or PatternParameters()
    
    def generate_pattern(self, data: CellData, n_cells: int, 
                        pattern_type: PatternType) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成人工空間模式
        
        Args:
            data: 原始 CellData 作為參考
            n_cells: 要生成的細胞數量
            pattern_type: 要生成的模式類型
            
        Returns:
            生成模式的 (x_coords, y_coords) 元組
        """
        if pattern_type == PatternType.ORIGINAL:
            # 返回原始數據的子集
            if n_cells >= len(data.x):
                return data.x, data.y
            indices = np.random.choice(len(data.x), n_cells, replace=False)
            return data.x[indices], data.y[indices]
        
        # 計算原始數據的最小包圍多邊形
        original_points = np.column_stack((data.x, data.y))
        bounding_polygon = compute_minimum_bounding_polygon(original_points, 
                                                          self.params.densify_k,
                                                          original_points)
        
        if pattern_type == PatternType.RANDOM:
            # 在多邊形範圍內隨機產生點
            return generate_random_points_in_polygon(bounding_polygon, n_cells)
        
        elif pattern_type == PatternType.REGULAR:
            # 在多邊形範圍內排列相互距離接近的點（交錯蜂巢格模式）
            return generate_regular_points_in_polygon(bounding_polygon, n_cells, 
                                                   original_points, self.params)
        
        elif pattern_type == PatternType.CLUSTER:
            # 在多邊形範圍內產生聚集模式的點
            return generate_cluster_points_in_polygon(bounding_polygon, n_cells, 
                                                   original_points, self.params)
        
        else:
            raise ValueError(f"未知的模式類型: {pattern_type}")

# 使用範例
if __name__ == "__main__":
    # 創建示例數據
    #np.random.seed(42)
    n_original = 200
    
    # 生成一些原始點（不規則分佈）
    theta = np.random.uniform(0, 2*np.pi, n_original)
    r = np.random.uniform(1, 5, n_original)
    x_orig = r * np.cos(theta) + np.random.normal(0, 0.5, n_original)
    y_orig = r * np.sin(theta) + np.random.normal(0, 0.5, n_original)
    
    # 創建自定義參數配置 (不設定則使用預設)
    '''
    params = PatternParameters(
        density_factor=1.2,        # 這個值會被動態計算覆蓋
        variance_min=0.05,         # 最小變異度
        variance_max=0.2,          # 最大變異度
        cluster_radius_factor=0.2, # 聚集半徑因子
        min_clusters=3,            # 最小聚集數
        max_clusters=36,           # 最大聚集數
        density_tolerance=0.1,     # 密度容差
        dbscan_eps=0.5,           # DBSCAN 聚類參數
        dbscan_min_samples=5      # DBSCAN 最小樣本數
    )
    '''
    
    data = CellData(x_orig, y_orig)
    generator = PatternGenerator()
    
    # 測試不同模式
    n_generate = n_original
    
    print("原始數據點數:", len(data.x))
    print("要生成的點數:", n_generate)
    
    # 計算包圍多邊形（使用細緻化）
    original_points = np.column_stack((data.x, data.y))
    polygon = compute_minimum_bounding_polygon(original_points, 
                                            densify_k=5,
                                            original_points=original_points)
    print(f"包圍多邊形頂點數: {len(polygon)}")
    
    # 計算原始點的中位數距離
    orig_median_dist = calculate_median_distance(original_points)
    print(f"原始點中位數距離: {orig_median_dist:.4f}")
    
    # 創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pattern Comparison', fontsize=16)
    
    # 繪製原始數據
    axes[0, 0].scatter(data.x, data.y, c='blue', alpha=0.6, label='Ori Pt.')
    # 閉合多邊形
    closed_polygon = np.vstack((polygon, polygon[0]))
    axes[0, 0].plot(closed_polygon[:, 0], closed_polygon[:, 1], 'r--', label='ConvexHull')
    axes[0, 0].set_title(f'Original\nMedian Dist: {orig_median_dist:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 生成並繪製不同模式
    patterns = [PatternType.RANDOM, PatternType.REGULAR, PatternType.CLUSTER]
    titles = ['Random', 'Regular', 'Cluster']
    positions = [(0, 1), (1, 0), (1, 1)]
    
    for pattern_type, title, pos in zip(patterns, titles, positions):
        x_new, y_new = generator.generate_pattern(data, n_generate, pattern_type)
        generated_points = np.column_stack((x_new, y_new))
        
        # 評估生成結果
        wasserstein_sim = evaluate_spatial_similarity(original_points, generated_points)
        median_sim = evaluate_median_similarity(original_points, generated_points)
        gen_median_dist = calculate_median_distance(generated_points)
        
        print(f"\n{pattern_type.value} 模式:")
        print(f"生成了 {len(x_new)} 個點")
        print(f"中位數距離: {gen_median_dist:.4f}")
        print(f"Wasserstein 距離: {wasserstein_sim:.4f}")
        print(f"中位數相似度: {median_sim:.4f}")
        
        axes[pos].scatter(x_new, y_new, c='green', alpha=0.6, label='Gen Pt')
        # 閉合多邊形
        closed_polygon = np.vstack((polygon, polygon[0]))
        axes[pos].plot(closed_polygon[:, 0], closed_polygon[:, 1], 'r--', label='ConvexHull')
        axes[pos].set_title(f"{title}\nW-Dist: {wasserstein_sim:.4f}\nM-Sim: {median_sim:.4f}\nMed-Dist: {gen_median_dist:.4f}")
        axes[pos].legend()
        axes[pos].grid(True)
    
    plt.tight_layout()
    plt.show()
