'''
1. 通过osmnx构建路网和图。
2. 输入起点终点坐标
2.1 通过osmnx获取路网的起点终点最近的坐标
2.2 调用高德api获取几条不同的基本路径的polyline
3. 通过算法以高德的基本路径为firework,进行周围路径探索(若干sparks)
4. 将firework和sparks的路径长度进行比较,选择最短的路径作为当前的best path
4pro. firework和sparks涉及到的路段不用距离做衡量,而是调用高德api获取的该段的平均speed,距离/speed的时间作为权重。
5. 可视化,可以做一个html网页,支持点击输入起点终点,输出路径。
'''
# TODO: 路径和路况也可以输出
import osmnx as ox
from networkx import DiGraph, MultiDiGraph, is_isolate
import numpy as np
import requests
import time
import folium
import json
from IPython.display import display, HTML
import networkx as nx
from shapely.geometry import LineString, Point
from collections import deque
import matplotlib.pyplot as plt
import random
from flask import Flask, request, jsonify

def build_road_network(place_name="海淀区, 北京", simplify=True):
    """构建带权有向路网"""
    # 从OSM下载原始路网
    G = ox.graph_from_place(place_name, network_type='drive', simplify=simplify)  # MultiDiGraph

    # 排除掉 residential 属性的道路
    edges_to_keep = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if "highway" in data and not (
            isinstance(data["highway"], str) and data["highway"] == "residential" or
            isinstance(data["highway"], list) and "residential" in data["highway"]
        )
    ]
    G_major = G.edge_subgraph(edges_to_keep).copy()

    # 检查并移除孤立节点
    isolated_nodes = [node for node in G_major.nodes if is_isolate(G_major, node)]
    if isolated_nodes:
        print(f"发现孤立节点：{len(isolated_nodes)} 个，正在移除...")
        G_major.remove_nodes_from(isolated_nodes)
    else:
        print("未发现孤立节点")

    # 转换为有向图
    digraph = nx.DiGraph(G_major)
    return G_major, digraph
    




    # 转换为带权有向图
    # weighted_G = DiGraph()
    # weighted_G = MultiDiGraph()
    # weighted_G.graph["crs"] = G.graph["crs"]  # 复制原始图的 CRS 属性
    
    # for node, data in G.nodes(data=True):
    #     # 复制节点的属性，包括 x 和 y 坐标
    #     weighted_G.add_node(node, **data)

    # for u, v, data in G.edges(data=True):
    #     # 计算路段长度作为权重
    #     length = data.get('length', 100)  # 默认长度100米
    #     weighted_G.add_edge(u, v, weight=length)
        
    #     # 要处理单向道路吗
        
    
    # # 检查并移除孤立节点
    # isolated_nodes = [node for node in weighted_G.nodes if is_isolate(weighted_G, node)]
    # if isolated_nodes:
    #     print(f"发现孤立节点：{len(isolated_nodes)}个")
    #     weighted_G.remove_nodes_from(isolated_nodes)
    # else:
    #     print("未发现孤立节点")
    
    # return G_major, digraph

def get_path_road_names(G, paths):
    """获取路径经过的所有道路名称"""
    road_nodes = {}
    
    for i in range(len(paths) - 1):
        u, v = paths[i], paths[i + 1]

        # 获取 u 与 v 之间的所有边的数据
        edges = G.get_edge_data(u, v)
        if edges is None:
            print(f"节点 {u} 和 {v} 之间没有边。")
            continue

        # 遍历所有边
        for key, edge_data in edges.items():
            # 获取边的名称
            name = edge_data.get('name', None)
            if name:
                # 如果名称是列表，逐个处理
                if isinstance(name, list):
                    for n in name:
                        if n not in road_nodes:
                            road_nodes[n] = []
                        if u not in road_nodes[n]:
                            road_nodes[n].append(u)
                        if v not in road_nodes[n]:
                            road_nodes[n].append(v)
                else:
                    if name not in road_nodes:
                        road_nodes[name] = []
                    if u not in road_nodes[name]:
                        road_nodes[name].append(u)
                    if v not in road_nodes[name]:
                        road_nodes[name].append(v)
    
    return road_nodes

def filter_major_roads(G):
    # 将图转换为 GeoDataFrame
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # 主要道路类别
    major_highways = {"motorway", "trunk", "primary", "secondary", "tertiary"}

    # 过滤边，保留主要道路
    gdf_edges_filtered = gdf_edges[gdf_edges["highway"].apply(lambda x: isinstance(x, list) and any(h in major_highways for h in x) or x in major_highways)]

    # 重新构建图
    G_filtered = ox.graph_from_gdfs(gdf_nodes, gdf_edges_filtered)
    
    return G_filtered

def locate_nodes(network, origins, destinations):
    """将经纬度坐标映射到最近路网节点"""
    nodes = []
    for point in origins + destinations:
        node = ox.distance.nearest_nodes(network, point[0], point[1])
        nodes.append(node)
    return nodes[0], nodes[1]

def get_amap_routes(origin, destination, amap_key):
    """调用高德地图API获取路径规划"""
    url = "https://restapi.amap.com/v3/direction/driving"
    
    params = {
        "key": amap_key,
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "strategy": 11,  # 返回三个结果包含：时间最短；距离最短；躲避拥堵 
        "extensions": "all",
        "output": "json",
        "alternatives": "3"  # 获取多条路径
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["status"] == "1":
        routes = []
        for route in data["route"]["paths"]:
            polyline = []
            for step in route["steps"]:
                # 解析每一步的polyline
                points = step["polyline"].split(";")
                for point in points:
                    lng, lat = point.split(",")
                    polyline.append((float(lng), float(lat)))
            
            # 获取路径信息
            route_info = {
                "polyline": polyline,
                "distance": float(route["distance"]),
                "duration": float(route["duration"]),
                "tolls": route.get("tolls", 0),
                "traffic_lights": route.get("traffic_lights", 0)
            }
            routes.append(route_info)
        return routes
    else:
        print(f"高德API请求失败: {data.get('info', '未知错误')}")
        return []

def find_path_on_network(G, start_node, end_node, polyline):
    """根据高德polyline在OSM路网上找到对应路径"""
    path = []
    current_node = start_node
    
    # 将polyline转换为OSM节点序列
    polyline_nodes = []
    for point in polyline:
        node = ox.distance.nearest_nodes(G, point[0], point[1])
        polyline_nodes.append(node)
    
    # 去除重复节点
    polyline_nodes = [polyline_nodes[i] for i in range(len(polyline_nodes)) 
                     if i == 0 or polyline_nodes[i] != polyline_nodes[i-1]]
    
    # 确保起点和终点正确
    if polyline_nodes[0] != start_node:
        polyline_nodes.insert(0, start_node)
    if polyline_nodes[-1] != end_node:
        polyline_nodes.append(end_node)
    
    # 在每对连续节点之间找到最短路径
    for i in range(len(polyline_nodes) - 1):
        try:
            segment = nx.shortest_path(G, 
                                      source=polyline_nodes[i], 
                                      target=polyline_nodes[i+1], 
                                      weight='length')
            # 添加路径但避免重复节点
            if path and path[-1] == segment[0]:
                path.extend(segment[1:])
            else:
                path.extend(segment)
        except nx.NetworkXNoPath:
            print(f"无法找到从节点 {polyline_nodes[i]} 到节点 {polyline_nodes[i+1]} 的路径")
    
    return path


def generate_sparks(G, firework_path, num_sparks=5, min_distance=5, max_distance=15, min_angle=30, max_angle=150):
    sparks = []
    
    # 计算节点之间的地理距离
    def calculate_distance(node1, node2):
        lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
        lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
        return ox.distance.great_circle(lat1, lon1, lat2, lon2)
    
    # 计算两条边的夹角
    def calculate_angle(G, branch_node, nextnode, startnode, endnode):
        # 获取节点坐标
        lat1, lon1 = G.nodes[branch_node]['y'], G.nodes[branch_node]['x']
        lat2, lon2 = G.nodes[nextnode]['y'], G.nodes[nextnode]['x']
        lat3, lon3 = G.nodes[startnode]['y'], G.nodes[startnode]['x']
        lat4, lon4 = G.nodes[endnode]['y'], G.nodes[endnode]['x']

        # 计算向量
        vec1 = np.array([lon2 - lon1, lat2 - lat1])  # branch_node -> nextnode 向量
        vec2 = np.array([lon4 - lon3, lat4 - lat3])  # startnode -> endnode 向量

        # 计算角度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免计算误差

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg
    def adaptive_distance_range(firework_path, branch_point_idx, min_distance_max, max_distance_max, decay_factor=0.05):
        """
        根据 branch_point_idx 调整 min_distance 和 max_distance，使其随 index 增大而递减

        :param firework_path: 路径点列表
        :param branch_point_idx: 分叉点在路径中的索引
        :param min_distance_max: 初始最小距离
        :param max_distance_max: 初始最大距离
        :param decay_factor: 衰减系数，控制 min/max_distance 的递减速度
        :return: 计算后的 (min_distance, max_distance)
        """
        total_points = len(firework_path)  # 路径总点数
        norm_idx = branch_point_idx / total_points  # 归一化索引 (0~1)

        # **线性衰减**
        # min_distance = min_distance_max * (1 - norm_idx)
        # max_distance = max_distance_max * (1 - norm_idx)

        # **指数衰减**（减少过快时可以调整 decay_factor）
        min_distance = min_distance_max * np.exp(-decay_factor * branch_point_idx)
        max_distance = max_distance_max * np.exp(-decay_factor * branch_point_idx)

        return max(min_distance, 0), max(max_distance, 0)  # 避免负值
    # 找出一定距离范围内的节点
    def find_nodes_in_distance_range(center_node, min_dist, max_dist):
        candidates = []
        center_point = (G.nodes[center_node]['y'], G.nodes[center_node]['x'])
        
        # 使用NetworkX的ego_graph来获得一定范围内的节点
        radius_graph_max = nx.ego_graph(G, center_node, radius=max_dist)  # 粗略估计，后面会精确计算
        radius_graph_min = nx.ego_graph(G, center_node, radius=min_dist)

        for node in set(radius_graph_max.nodes()) - set(radius_graph_min.nodes()):
            # if node != center_node:
            #     dist = calculate_distance(center_node, node)
            #     if min_dist <= dist <= max_dist:
            candidates.append(node)
        
        return candidates
    
    for i in range(num_sparks):
        # 随机选择是从起点出发还是从终点出发
        # if np.random.rand() > 0.5:
        # 从起点出发
        source = firework_path[0]
        target = firework_path[-1]
        direction = 'forward'
        # else:
        #     # 从终点出发
        #     source = firework_path[-1]
        #     target = firework_path[0]
        #     direction = 'backward'
        
        # 从路径中随机选择一个节点作为分叉点
        if i < 10:
            branch_point_idx = 0
        elif i < 20:
            branch_point_idx = len(firework_path) - 1
        else:
            if len(firework_path) > 2:
                branch_point_idx = np.random.randint(0, len(firework_path) - 1)
            else:
                branch_point_idx = 0
        
        branch_point = firework_path[branch_point_idx]
        
        # 如果不是起点，获取前一个节点；如果是起点，使用下一个节点
        # if branch_point_idx > 0:
        #     prev_node = firework_path[branch_point_idx - 1]
        # else:
        #     prev_node = firework_path[branch_point_idx + 1]

        # 我需要找next_node
        # if branch_point_idx < len(firework_path) - 1 and direction == 'forward':
        #     next_node = firework_path[branch_point_idx + 1]
        # elif branch_point_idx > 0 and direction == 'backward':
        #     next_node = firework_path[branch_point_idx - 1]
        # elif branch_point_idx == 0:
        #     next_node = firework_path[branch_point_idx + 1]
        # else:
        #     next_node = firework_path[branch_point_idx - 1]
        
        # 获取符合距离条件的候选节点
        min_distance_, max_distance_ = adaptive_distance_range(firework_path, branch_point_idx, min_distance, max_distance)
        print(f"第 {i+1} 条spark: 分叉点索引 {branch_point_idx}, 最小距离 {min_distance_:.2f}, 最大距离 {max_distance_:.2f}")
        candidates = find_nodes_in_distance_range(branch_point, min_distance_, max_distance_)
        
        if not candidates:
            print(f"没有找到符合条件的候选节点，跳过第 {i+1} 条spark")
            continue
        
        # 筛选符合角度条件的候选节点
        valid_candidates = []
        for node in candidates:
            angle = calculate_angle(G, branch_point, node, source, target)
            if 0 <= angle <= 60:
                # print(f"节点 {node} 的角度为 {angle:.2f}°，符合条件")
                valid_candidates.append(node)
        
        if not valid_candidates:
            continue
        
        # 创建一个候选节点的副本，用于随机选择
        remaining_candidates = valid_candidates[:]

        while remaining_candidates:
            # 随机选择一个符合条件的新节点
            new_node = random.choice(remaining_candidates)
            remaining_candidates.remove(new_node)  # 从候选列表中移除已尝试的节点

            # 寻找从分叉点到新节点的路径
            try:
                branch_path = nx.shortest_path(G, branch_point, new_node, weight='length')
            except nx.NetworkXNoPath:
                continue  # 如果没有路径，尝试下一个节点

            # 寻找从新节点到目标的路径
            try:
                remaining_path = nx.shortest_path(G, new_node, target, weight='length')
            except nx.NetworkXNoPath:
                continue  # 如果没有路径，尝试下一个节点

            # 检查有没有走回头路
            remaining_slice = remaining_path[1:5] if len(remaining_path) > 5 else remaining_path[1:]
            if set(branch_path[1:-1]) & set(remaining_slice):
                # 如果路径有交集，尝试下一个节点
                continue

            # 如果找到符合条件的路径，退出循环
            break
        else:
            # 如果所有候选节点都尝试过且没有找到符合条件的路径
            print(f"无法找到符合条件的路径，跳过第 {i+1} 条 spark")
            continue

        # 构建完整路径
        if direction == 'forward':
            # 分叉点前面的路径 + 分叉路径 + 剩余路径(除去新节点，避免重复)
            new_path = firework_path[:branch_point_idx+1] + branch_path[1:] + remaining_path[1:]
        else:
            # 反向：从终点到分叉点的路径 + 分叉路径 + 从新节点到起点的路径(除去新节点)
            new_path = firework_path[:branch_point_idx+1] + branch_path[1:] + remaining_path[1:]
            new_path.reverse()  # 需要反转以保持从起点到终点的顺序
            
        sparks.append(new_path)
    
    return sparks

def calculate_path_cost(G, path, use_speed=False, amap_key=None):
    """计算路径的总成本（距离或时间）"""
    if not path:
        return float('inf')
    
    total_cost = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            if use_speed and amap_key:
                # 获取节点的经纬度
                u_lon, u_lat = G.nodes[u]['x'], G.nodes[u]['y']
                v_lon, v_lat = G.nodes[v]['x'], G.nodes[v]['y']
                
                # 调用高德API获取路段的平均速度
                avg_speed = get_segment_speed(amap_key, (u_lon, u_lat), (v_lon, v_lat))
                
                # 计算时间成本 = 距离/速度
                distance = G[u][v][0]['length']
                time_cost = distance / max(avg_speed, 1)  # 避免除以零
                total_cost += time_cost
            else:
                # 使用距离作为权重
                total_cost += G[u][v][0]['length']
        else:
            # 如果边不存在，给予一个很大的惩罚
            total_cost += 10000
    
    return total_cost

def get_segment_speed(amap_key, origin, destination):
    """调用高德API获取路段的平均速度"""
    url = "https://restapi.amap.com/v3/direction/driving"
    
    params = {
        "key": amap_key,
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "strategy": 0,
        "output": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["status"] == "1" and data["route"]["paths"]:
            # 获取距离和时间
            distance = float(data["route"]["paths"][0]["distance"])  # 米
            duration = float(data["route"]["paths"][0]["duration"])  # 秒
            
            # 计算平均速度 (米/秒)
            avg_speed = distance / max(duration, 1)  # 避免除以零
            return avg_speed
        else:
            return 10  # 默认速度 10 m/s (36 km/h)
    except:
        return 10  # 请求失败时的默认速度

def find_best_path(G, diG, origin, destination, amap_key, use_speed=False, num_sparks=10, really_use_api=False):
    """找到从起点到终点的最佳路径"""
    # 1. 找到OSM网络中最近的起点和终点节点
    start_node, end_node = locate_nodes(G, [origin], [destination])
    # 增加两个节点到路网并分别连接他们和startnode，endnode



    # 2. 获取高德API的基本路径
    amap_routes = get_amap_routes(origin, destination, amap_key)
    
    if not amap_routes:
        print("无法从高德API获取路径，使用OSM默认最短路径")
    
    # 3. 将高德路径转换为OSM网络上的路径（firework）
    fireworks = []
    for route in amap_routes:
        firework_path = find_path_on_network(G, start_node, end_node, route["polyline"])
        if firework_path:
            fireworks.append(firework_path)
    
    if not fireworks:
        print("无法将高德路径映射到OSM网络上，使用OSM默认最短路径")
        try:
            default_path = nx.shortest_path(G, start_node, end_node, weight='length')
            print(default_path)
            fireworks.append(default_path)
            # return default_path, G, None
        except nx.NetworkXNoPath:
            print("无法找到从起点到终点的路径")
            return None, G, None, None
    
    # 4. 生成多个spark路径
    all_paths = []
    for firework in fireworks:
        all_paths.append(firework)
        sparks = generate_sparks(diG, firework, num_sparks)
        print(f"生成 {len(sparks)} 条spark路径")
        all_paths.extend(sparks)
    
    # 5. 计算所有路径的成本并找到最佳路径
    best_path = None
    lowest_cost = float('inf')
    road_info_dict = {}
    if not use_speed:
        for path in all_paths:
            cost = calculate_path_cost(G, path, use_speed, amap_key)
            if cost < lowest_cost:
                lowest_cost = cost
                best_path = path
    else:
        # highway 类型对应的默认速度（单位：km/h）
        highway_default_speeds = {
            "motorway": 80,           # 高速公路，降低到 80 km/h
            "motorway_link": 50,      # 高速公路连接路段，降低到 50 km/h
            "trunk": 70,              # 主干道，降低到 70 km/h
            "trunk_link": 50,         # 主干道连接路段，降低到 50 km/h
            "primary": 50,            # 主要道路，降低到 50 km/h
            "primary_link": 40,       # 主要道路连接路段，降低到 40 km/h
            "secondary": 40,          # 次要道路，降低到 40 km/h
            "secondary_link": 30,     # 次要道路连接路段，降低到 30 km/h
            "tertiary": 30,           # 第三级道路，降低到 30 km/h
            "tertiary_link": 20,      # 第三级道路连接路段，降低到 20 km/h
            "unclassified": 20,       # 未分类道路，降低到 20 km/h
            "residential": 20,        # 居民区道路，降低到 20 km/h
            "living_street": 10,      # 生活街区，保持 10 km/h
            "road": 15,               # 普通道路，降低到 15 km/h
            "scramble": 5,            # 混合区域，保持 5 km/h
        }
        all_paths_road_names = []

        road_nodes_all = {}
        for path in all_paths:
            road_nodes = get_path_road_names(G, path)
            # 合并入road_nodes_all
            for road_name, nodes in road_nodes.items():
                if road_name in road_nodes_all:
                    road_nodes_all[road_name].extend(nodes)
                else:
                    road_nodes_all[road_name] = nodes
            road_names = list(road_nodes.keys())
            all_paths_road_names.append(road_names)

        # set一下road_nodes_all的value
        for road_name, nodes in road_nodes_all.items():
            road_nodes_all[road_name] = list(set(nodes))

        unique_road_names = set(road_name for road_names in all_paths_road_names for road_name in road_names)
        print(f"所有路径经过的道路名称：{unique_road_names}")

        if really_use_api:
            ak = "ZJcr2bfcgSCSxSDw2HdpodSPrC2CtvDd"
            city = "北京市"
            # road_info_dict = {}

            for road_name in unique_road_names:
                params = {
                    "road_name": road_name,
                    "city": city,
                    "ak": ak,
                }
                response = requests.get(url="https://api.map.baidu.com/traffic/v1/road", params=params)
                if response.status_code == 200:
                    data = response.json()
                    print(f"查询道路: {road_name}, 返回数据: {data}")  # 输出API响应数据
                    
                    if data.get("status") == 0:
                        description = data.get("description", "")
                        road_traffic = data.get("road_traffic", [])
                        if road_traffic:
                            congestion_sections = road_traffic[0].get("congestion_sections", [])
                            if congestion_sections:
                                avg_speed = sum(section.get("speed", 0) for section in congestion_sections) / len(congestion_sections)
                                road_info_dict[road_name] = {"description": description, "speed": avg_speed, "node_coo": 
                                                             [(G.nodes[node].get('y', 'Unknown'), G.nodes[node].get('x', 'Unknown')) for node in road_nodes_all[road_name]]}
                            else:
                                print(f"没有拥堵段数据: {road_name}")
                        else:
                            print(f"没有路段信息: {road_name}")
                    else:
                        print(f"API返回状态不为0, 无法获取 {road_name} 的数据")
                else:
                    print(f"API请求失败: {road_name}")
                time.sleep(1)  # 避免过于频繁的请求

            # 保存 road_info_dict 到本地文件
            with open('road_info_dict.json', 'w', encoding='utf-8') as f:
                json.dump(road_info_dict, f, ensure_ascii=False, indent=4)
        else:
            # 从本地文件加载 road_info_dict
            with open('road_info_dict.json', 'r', encoding='utf-8') as f:
                road_info_dict = json.load(f)
       
        path_costs = []

        for path, road_names in zip(all_paths, all_paths_road_names):
            total_time = 0.0
            idx = 0
            lengths = []
            times = []
            previous_name = None    

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = G.get_edge_data(u, v)
                for key, edge_data in edge.items():
                    name = edge_data.get('name', None)
                    length = edge_data.get('length', 0)  # 边的长度，单位：米
                    if isinstance(name, list):
                        name = name[0] if name else None  # 如果列表为空，设置为 None
                    if isinstance(length, list):
                        length = length[0] if length else 0
                    # 如过name是None，follow上一个name
                    if name is None:
                        name = previous_name
                    else:
                        previous_name = name
                    # print(f"当前道路名称: {name}")
                    if name not in road_info_dict or name is None:
                        highway = edge_data.get('highway', None)
                        if isinstance(highway, list):
                            highway = highway[0] if highway else None
                        
                        speed = highway_default_speeds.get(highway)  # 默认速度20km/h
                        time_ = length / (speed * 1000 / 3600)  # 转换为秒
                        times.append(time_)
                    else:
                        # 如果有速度信息，使用速度计算时间
                        if road_info_dict[name]["speed"] > 0:
                            time_ = length / (road_info_dict[name]["speed"] * 1000 / 3600)
                            times.append(time_)
            #         lengths.append(length)
            #         new_idx = road_names.index(name) if name in road_names else -1
            #         if new_idx == idx + 1:
            #             idx = new_idx
            #             sum_length = sum(lengths[:-1])
            #             lengths = [lengths[-1]]
            #             # 判断是否有速度信息
            #             if road_names[idx-1] in road_info_dict:
            #                 speed = road_info_dict[road_names[idx-1]]["speed"] * 1000 / 3600
            #                 if speed > 0:
            #                     time_ = sum_length / speed
            #                     times.append(time_)
                        
            # # 计算最后一段道路的时间
            # if lengths:
            #     sum_length = sum(lengths)
            #     if road_names[idx] in road_info_dict:
            #         speed = road_info_dict[road_names[idx]]["speed"] * 1000 / 3600  # 转换为米/秒
            #         if speed > 0:
            #             time_ = sum_length / speed
            #             times.append(time_)
            total_time = sum(times)
            path_costs.append(total_time)
            if total_time < lowest_cost:
                lowest_cost = total_time
                best_path = path
        # print(f"所有路径的平均时间：{path_costs}")
    # 6. 输出结果    
    print(f"共计算了 {len(all_paths)} 条路径，找到成本为 {lowest_cost} 的最佳路径")
    
    return best_path, G, fireworks, all_paths, road_info_dict



def create_interactive_map(G, best_path, fireworks=None, origin=None, destination=None):
    """创建交互式地图"""
    # 创建基础地图
    if origin and destination:
        center_lat = (origin[1] + destination[1]) / 2
        center_lon = (origin[0] + destination[0]) / 2
    else:
        center_lat, center_lon = 39.9042, 116.4074  # 默认北京中心
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # 添加高德提供的基本路径（fireworks）
    if fireworks:
        for i, firework in enumerate(fireworks):
            path_coords = []
            for node in firework:
                lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
                path_coords.append([lat, lon])
            
            folium.PolyLine(
                path_coords,
                color='blue',
                weight=3,
                opacity=0.5,
                tooltip=f'Firework Path {i+1}'
            ).add_to(m)
    
    # 添加最优路径
    if best_path:
        path_coords = []
        for node in best_path:
            lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
            path_coords.append([lat, lon])
        
        folium.PolyLine(
            path_coords,
            color='red',
            weight=5,
            opacity=0.8,
            tooltip='Best Path'
        ).add_to(m)
    
    # 标记起点和终点
    if origin and destination:
        # 添加起点标记
        folium.Marker(
            location=[origin[1], origin[0]],
            popup='Origin',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # 添加终点标记
        folium.Marker(
            location=[destination[1], destination[0]],
            popup='Destination',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
    
    return m

# 主函数
def main():
    # 设置高德地图 API Key
    amap_key = "20eab95559788e3ad3f71cf404c60821h"  # 需要替换为实际的API Key
    
    # 构建路网
    print("正在构建路网...")
    G, diG = build_road_network("海淀区, 北京", simplify=True)
    print(f"路网构建完成，包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")
    
    # 设置示例起点和终点坐标（经度，纬度）
    origin = (116.34899139404298, 39.95976154261228)  # 北京邮电大学西门
    destination = (116.30986976728312, 39.99037781712655)  
    # origin=(39.93093245403986, 116.3668441772461)
    # destination=(39.93942226632577, 116.43851280212404)
    # 查找最佳路径
    print("正在计算最佳路径...")
    best_path, updated_G, fireworks, all_paths, road_info_dict = find_best_path(G, diG, origin, destination, amap_key, use_speed=True, num_sparks=150, really_use_api=False)
    
    if best_path:
        print("找到最佳路径！")
        
        # # 可视化路径
        # print("生成可视化结果...")
        # fig = visualize_paths(updated_G, best_path, fireworks, origin, destination)
        # plt.savefig("best_route.png")wegame
        # print("静态可视化结果已保存为 'best_route.png'")
        
        # 创建交互式地图
        m = create_interactive_map(updated_G, best_path, all_paths, origin, destination)
        m.save("interactive_map.html")
        print("交互式地图已保存为 'interactive_map.html'")
        
    else:
        print("无法找到从起点到终点的路径")

if __name__ == "__main__":
    main()


# def find_path_on_network(G, start_node, end_node, polyline, coord_threshold=0.00005):
#     """根据高德polyline在OSM路网上找到对应路径，必要时添加新节点
    
#     参数:
#         G: OSM网络图
#         start_node: 起点节点ID
#         end_node: 终点节点ID
#         polyline: 高德返回的路径坐标列表 [(lng1, lat1), (lng2, lat2), ...]
#         coord_threshold: 坐标差值阈值，默认0.00005度
#     """
#     # 创建一个图的副本，避免修改原图
#     modified_G = G.copy()
    
#     # 获取起点和终点的坐标
#     start_coords = (G.nodes[start_node]['x'], G.nodes[start_node]['y'])
#     end_coords = (G.nodes[end_node]['x'], G.nodes[end_node]['y'])
    
#     # 确保polyline包含起点和终点附近的点
#     if coords_distance(start_coords, polyline[0]) > coord_threshold:
#         polyline.insert(0, start_coords)
#     if coords_distance(end_coords, polyline[-1]) > coord_threshold:
#         polyline.append(end_coords)
    
#     # 创建一个字典，存储已有节点的坐标到节点ID的映射
#     coords_to_node = {(G.nodes[n]['x'], G.nodes[n]['y']): n for n in G.nodes()}
    
#     # 映射polyline点到OSM节点或创建新节点
#     polyline_nodes = []
#     new_node_id = max(G.nodes()) + 1  # 新节点ID从最大节点ID开始
    
#     for point in polyline:
#         # 检查是否有足够近的现有节点
#         found_existing = False
        
#         for coords, node_id in coords_to_node.items():
#             if coords_distance(point, coords) <= coord_threshold:
#                 polyline_nodes.append(node_id)
#                 found_existing = True
#                 break
        
#         if not found_existing:
#             # 创建新节点
#             modified_G.add_node(
#                 new_node_id,
#                 x=point[0],
#                 y=point[1],
#                 osmid=f"amap_{new_node_id}"
#             )
#             polyline_nodes.append(new_node_id)
#             # 更新坐标映射
#             coords_to_node[(point[0], point[1])] = new_node_id
#             new_node_id += 1
    
#     # 连接polyline节点
#     for i in range(len(polyline_nodes) - 1):
#         node1 = polyline_nodes[i]
#         node2 = polyline_nodes[i + 1]
        
#         if not modified_G.has_edge(node1, node2):
#             # 计算简单欧氏距离作为权重（也可以用真实距离代替）
#             point1 = (modified_G.nodes[node1]['x'], modified_G.nodes[node1]['y'])
#             point2 = (modified_G.nodes[node2]['x'], modified_G.nodes[node2]['y'])
#             # 这里计算的是简化的距离，用于权重
#             dx = (point1[0] - point2[0]) * 111320 * np.cos(np.radians((point1[1] + point2[1]) / 2))
#             dy = (point1[1] - point2[1]) * 110540
#             distance = np.sqrt(dx**2 + dy**2)  # 单位：米
            
#             # 添加双向边
#             modified_G.add_edge(node1, node2, weight=distance, length=distance)
#             modified_G.add_edge(node2, node1, weight=distance, length=distance)
    
#     # 确保起点和终点与路径相连
#     if start_node != polyline_nodes[0] and not modified_G.has_edge(start_node, polyline_nodes[0]):
#         point1 = (modified_G.nodes[start_node]['x'], modified_G.nodes[start_node]['y'])
#         point2 = (modified_G.nodes[polyline_nodes[0]]['x'], modified_G.nodes[polyline_nodes[0]]['y'])
#         dx = (point1[0] - point2[0]) * 111320 * np.cos(np.radians((point1[1] + point2[1]) / 2))
#         dy = (point1[1] - point2[1]) * 110540
#         distance = np.sqrt(dx**2 + dy**2)
#         modified_G.add_edge(start_node, polyline_nodes[0], weight=distance, length=distance)
#         modified_G.add_edge(polyline_nodes[0], start_node, weight=distance, length=distance)
    
#     if end_node != polyline_nodes[-1] and not modified_G.has_edge(end_node, polyline_nodes[-1]):
#         point1 = (modified_G.nodes[end_node]['x'], modified_G.nodes[end_node]['y'])
#         point2 = (modified_G.nodes[polyline_nodes[-1]]['x'], modified_G.nodes[polyline_nodes[-1]]['y'])
#         dx = (point1[0] - point2[0]) * 111320 * np.cos(np.radians((point1[1] + point2[1]) / 2))
#         dy = (point1[1] - point2[1]) * 110540
#         distance = np.sqrt(dx**2 + dy**2)
#         modified_G.add_edge(end_node, polyline_nodes[-1], weight=distance, length=distance)
#         modified_G.add_edge(polyline_nodes[-1], end_node, weight=distance, length=distance)
    
#     # 尝试找到完整路径
#     try:
#         path = nx.shortest_path(modified_G, source=start_node, target=end_node, weight='weight')
#         return path, modified_G
#     except nx.NetworkXNoPath:
#         print(f"即使添加了polyline节点，仍无法找到从起点到终点的路径")
#         return None, modified_G

# def coords_distance(point1, point2):
#     """计算两个坐标点之间的简单欧氏距离（经纬度差值）"""
#     # 这是一个简化版本，只计算原始坐标差值的欧氏距离
#     # 并没有考虑实际地理距离
#     lng1, lat1 = point1
#     lng2, lat2 = point2
    
#     # 简单的欧氏距离（经纬度差值的平方和的平方根）
#     # 在小范围内可以作为近似
#     return np.sqrt((lng1 - lng2)**2 + (lat1 - lat2)**2)

# def visualize_paths(G, best_path, fireworks=None, origin=None, destination=None):
#     """可视化路径"""
#     fig, ax = plt.subplots(figsize=(15, 15))
    
#     # 绘制基本路网
#     ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, edge_color='grey')
    
#     # 绘制高德提供的基本路径（fireworks）
#     if fireworks:
#         for i, firework in enumerate(fireworks):
#             edges = [(firework[i], firework[i+1]) for i in range(len(firework)-1)]
#             color = 'blue'
#             ox.plot_graph_routes(G, firework, ax=ax, route_color=color, route_linewidth=1, route_alpha=0.5)
    
#     # 绘制最优路径
#     if best_path:
#         ox.plot_graph_route(G, best_path, ax=ax, route_color='red', route_linewidth=2)
    
#     # 标记起点和终点
#     if origin and destination:
#         # 绘制起点
#         ax.scatter(origin[0], origin[1], c='green', s=100, marker='^', zorder=5)
#         ax.annotate('Origin', (origin[0], origin[1]), fontsize=12)
        
#         # 绘制终点
#         ax.scatter(destination[0], destination[1], c='red', s=100, marker='*', zorder=5)
#         ax.annotate('Destination', (destination[0], destination[1]), fontsize=12)
    
#     plt.tight_layout()
#     return fig