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

import osmnx as ox
from networkx import DiGraph, MultiDiGraph, is_isolate
import numpy as np
import requests
import folium
import json
from IPython.display import display, HTML
import networkx as nx
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import random

def build_road_network(place_name="海淀区, 北京", simplify=True):
    """构建带权有向路网"""
    # 从OSM下载原始路网
    G = ox.graph_from_place(place_name, network_type='drive', simplify=simplify)
    
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
        
    #     # 处理双向道路
    #     if not data.get('oneway', False):
    #         weighted_G.add_edge(v, u, weight=length)
    
    # # 检查并移除孤立节点
    # isolated_nodes = [node for node in weighted_G.nodes if is_isolate(weighted_G, node)]
    # if isolated_nodes:
    #     print(f"发现孤立节点：{len(isolated_nodes)}个")
    #     weighted_G.remove_nodes_from(isolated_nodes)
    # else:
    #     print("未发现孤立节点")
    
    return G

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
                                      weight='weight')
            # 添加路径但避免重复节点
            if path and path[-1] == segment[0]:
                path.extend(segment[1:])
            else:
                path.extend(segment)
        except nx.NetworkXNoPath:
            print(f"无法找到从节点 {polyline_nodes[i]} 到节点 {polyline_nodes[i+1]} 的路径")
    
    return path


def generate_sparks(G, firework_path, num_sparks=5, deviation=0.3):
    """根据firework路径生成多个spark路径"""
    sparks = []
    
    # 获取firework路径的节点总数
    path_length = len(firework_path)
    
    for _ in range(num_sparks):
        # 随机选择一个分叉点和重新加入点
        fork_idx = random.randint(int(path_length * 0.1), int(path_length * 0.4))
        rejoin_idx = random.randint(fork_idx + int(path_length * 0.1), min(fork_idx + int(path_length * 0.5), path_length - 1))
        
        fork_node = firework_path[fork_idx]
        rejoin_node = firework_path[rejoin_idx]
        
        # 找到从分叉点出发的所有边
        neighbors = list(G.neighbors(fork_node))
        
        # 从邻居中排除已经在firework路径中的下一个节点
        if fork_idx + 1 < len(firework_path):
            next_node_in_path = firework_path[fork_idx + 1]
            if next_node_in_path in neighbors:
                neighbors.remove(next_node_in_path)
        
        if not neighbors:
            continue
        
        # 随机选择一个偏离方向
        deviation_node = random.choice(neighbors)
        
        # 尝试找到一条从偏离点到重新加入点的路径
        try:
            # 先找到从fork_node到deviation_node的路径
            deviation_path = [fork_node, deviation_node]
            
            # 再找到从deviation_node到rejoin_node的路径
            rejoin_path = nx.shortest_path(G, source=deviation_node, target=rejoin_node, weight='weight')
            
            # 组合路径：从起点到分叉点 + 偏离路径 + 从重新加入点到终点
            spark_path = firework_path[:fork_idx] + deviation_path[1:] + rejoin_path[1:] + firework_path[rejoin_idx+1:]
            
            sparks.append(spark_path)
        except nx.NetworkXNoPath:
            # 如果找不到路径，跳过这次尝试
            print(f"无法找到从 {deviation_node} 到 {rejoin_node} 的路径，跳过这条spark")
            continue
    
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
                distance = G[u][v]['weight']
                time_cost = distance / max(avg_speed, 1)  # 避免除以零
                total_cost += time_cost
            else:
                # 使用距离作为权重
                total_cost += G[u][v]['weight']
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

def find_best_path(G, origin, destination, amap_key, use_speed=False, num_sparks=5):
    """找到从起点到终点的最佳路径"""
    # 1. 找到OSM网络中最近的起点和终点节点
    start_node, end_node = locate_nodes(G, [origin], [destination])
    
    # 2. 获取高德API的基本路径
    amap_routes = get_amap_routes(origin, destination, amap_key)
    
    if not amap_routes:
        print("无法从高德API获取路径，使用OSM默认最短路径")
        try:
            default_path = nx.shortest_path(G, start_node, end_node, weight='weight')
            print(default_path)
            return default_path, G, None
        except nx.NetworkXNoPath:
            print("无法找到从起点到终点的路径")
            return None, G, None
    
    # 3. 将高德路径转换为OSM网络上的路径（firework）
    fireworks = []
    for route in amap_routes:
        firework_path = find_path_on_network(G, start_node, end_node, route["polyline"])
        if firework_path:
            fireworks.append(firework_path)
    
    if not fireworks:
        print("无法将高德路径映射到OSM网络上，使用OSM默认最短路径")
        try:
            default_path = nx.shortest_path(G, start_node, end_node, weight='weight')
            return default_path, G, None
        except nx.NetworkXNoPath:
            print("无法找到从起点到终点的路径")
            return None, G, None
    
    # 4. 生成多个spark路径
    all_paths = []
    for firework in fireworks:
        all_paths.append(firework)
        sparks = generate_sparks(G, firework, num_sparks)
        print(f"生成 {len(sparks)} 条spark路径")
        all_paths.extend(sparks)
    
    # 5. 计算所有路径的成本并找到最佳路径
    best_path = None
    lowest_cost = float('inf')
    
    for path in all_paths:
        cost = calculate_path_cost(G, path, use_speed, amap_key)
        if cost < lowest_cost:
            lowest_cost = cost
            best_path = path
    
    print(f"共计算了 {len(all_paths)} 条路径，找到成本为 {lowest_cost} 的最佳路径")
    
    return best_path, G, fireworks

def visualize_paths(G, best_path, fireworks=None, origin=None, destination=None):
    """可视化路径"""
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # 绘制基本路网
    ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, edge_color='grey')
    
    # 绘制高德提供的基本路径（fireworks）
    if fireworks:
        for i, firework in enumerate(fireworks):
            edges = [(firework[i], firework[i+1]) for i in range(len(firework)-1)]
            color = 'blue'
            ox.plot_graph_routes(G, firework, ax=ax, route_color=color, route_linewidth=1, route_alpha=0.5)
    
    # 绘制最优路径
    if best_path:
        ox.plot_graph_route(G, best_path, ax=ax, route_color='red', route_linewidth=2)
    
    # 标记起点和终点
    if origin and destination:
        # 绘制起点
        ax.scatter(origin[0], origin[1], c='green', s=100, marker='^', zorder=5)
        ax.annotate('Origin', (origin[0], origin[1]), fontsize=12)
        
        # 绘制终点
        ax.scatter(destination[0], destination[1], c='red', s=100, marker='*', zorder=5)
        ax.annotate('Destination', (destination[0], destination[1]), fontsize=12)
    
    plt.tight_layout()
    return fig

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

def create_route_planner_html():
    """创建一个简单的HTML路径规划器"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OSM & AMap Route Planner</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }
            #container {
                display: flex;
                height: 100vh;
            }
            #sidebar {
                width: 300px;
                padding: 20px;
                background-color: #f4f4f4;
                overflow-y: auto;
            }
            #map {
                flex-grow: 1;
                height: 100%;
            }
            .input-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input, select, button {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .path-info {
                margin-top: 20px;
                padding: 10px;
                background-color: #e9e9e9;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <div id="sidebar">
                <h2>路径规划</h2>
                <div class="input-group">
                    <label for="place">地区:</label>
                    <input type="text" id="place" value="海淀区, 北京">
                </div>
                <div class="input-group">
                    <label for="origin-lng">起点经度:</label>
                    <input type="text" id="origin-lng" placeholder="116.3452">
                </div>
                <div class="input-group">
                    <label for="origin-lat">起点纬度:</label>
                    <input type="text" id="origin-lat" placeholder="39.9789">
                </div>
                <div class="input-group">
                    <label for="dest-lng">终点经度:</label>
                    <input type="text" id="dest-lng" placeholder="116.4345">
                </div>
                <div class="input-group">
                    <label for="dest-lat">终点纬度:</label>
                    <input type="text" id="dest-lat" placeholder="39.9012">
                </div>
                <div class="input-group">
                    <label for="amap-key">高德地图 API Key:</label>
                    <input type="text" id="amap-key" placeholder="输入你的高德API Key">
                </div>
                <div class="input-group">
                    <label for="use-speed">使用速度作为权重:</label>
                    <select id="use-speed">
                        <option value="false">否 (使用距离)</option>
                        <option value="true">是 (使用时间)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="num-sparks">Spark 数量:</label>
                    <input type="number" id="num-sparks" value="5" min="1" max="20">
                </div>
                <button id="plan-route">规划路径</button>
                <button id="pick-on-map">在地图上选择点</button>
                
                <div class="path-info" id="path-info" style="display: none;">
                    <h3>路径信息</h3>
                    <div id="path-details"></div>
                </div>
            </div>
            <div id="map"></div>
        </div>

        <script>
            // 初始化地图
            var map = L.map('map').setView([39.9042, 116.4074], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
            
            // 用于存储标记和路径
            var markers = [];
            var paths = [];
            var pickingMode = false;
            var pickingOrigin = true;
            
            // 点击地图选择点
            document.getElementById('pick-on-map').addEventListener('click', function() {
                pickingMode = true;
                pickingOrigin = true;
                alert('点击地图选择起点');
            });
            
            map.on('click', function(e) {
                if (!pickingMode) return;
                
                var lat = e.latlng.lat;
                var lng = e.latlng.lng;
                
                if (pickingOrigin) {
                    document.getElementById('origin-lat').value = lat;
                    document.getElementById('origin-lng').value = lng;
                    alert('现在点击地图选择终点');
                    pickingOrigin = false;
                } else {
                    document.getElementById('dest-lat').value = lat;
                    document.getElementById('dest-lng').value = lng;
                    pickingMode = false;
                }
            });
            
            // 规划路径
            document.getElementById('plan-route').addEventListener('click', function() {
                var place = document.getElementById('place').value;
                var originLng = document.getElementById('origin-lng').value;
                var originLat = document.getElementById('origin-lat').value;
                var destLng = document.getElementById('dest-lng').value;
                var destLat = document.getElementById('dest-lat').value;
                var amapKey = document.getElementById('amap-key').value;
                var useSpeed = document.getElementById('use-speed').value === 'true';
                var numSparks = parseInt(document.getElementById('num-sparks').value);
                
                // 清除已有标记和路径
                clearMap();
                
                // 添加起点和终点标记
                addMarker([originLat, originLng], '起点', 'green');
                addMarker([destLat, destLng], '终点', 'red');
                
                // 设置地图视图
                map.fitBounds([
                    [originLat, originLng],
                    [destLat, destLng]
                ]);
                
                // 模拟路径规划
                simulateRoutePlanning(place, originLng, originLat, destLng, destLat, amapKey, useSpeed, numSparks);
            });
            
            function clearMap() {
                // 清除标记
                markers.forEach(function(marker) {
                    map.removeLayer(marker);
                });
                markers = [];
                
                // 清除路径
                paths.forEach(function(path) {
                    map.removeLayer(path);
                });
                paths = [];
            }
            
            function addMarker(latLng, title, color) {
                var icon = L.divIcon({
                    className: 'custom-div-icon',
                    html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%;"></div>`,
                    iconSize: [12, 12],
                    iconAnchor: [6, 6]
                });
                
                var marker = L.marker(latLng, {icon: icon}).addTo(map);
                marker.bindPopup(title);
                markers.push(marker);
                return marker;
            }
            
            function simulateRoutePlanning(place, originLng, originLat, destLng, destLat, amapKey, useSpeed, numSparks) {
                // 这里应该是调用后端API进行真实路径规划
                // 由于是前端示例，我们只模拟显示一些路径
                
                // 显示加载状态
                document.getElementById('path-details').innerHTML = '正在计算路径...';
                document.getElementById('path-info').style.display = 'block';
                
                // 模拟延迟
                setTimeout(function() {
                    // 模拟firework路径（蓝色）
                    var fireworkPath = generateSimulatedPath([originLat, originLng], [destLat, destLng], 0.001);
                    var firework = L.polyline(fireworkPath, {color: 'blue', weight: 3, opacity: 0.5}).addTo(map);
                    paths.push(firework);
                    
                    // 模拟生成sparks路径
                    for (var i = 0; i < numSparks; i++) {
                        var sparkPath = generateSimulatedPath([originLat, originLng], [destLat, destLng], 0.003);
                        var spark = L.polyline(sparkPath, {color: 'purple', weight: 2, opacity: 0.3}).addTo(map);
                        paths.push(spark);
                    }
                    
                    // 模拟最优路径（红色）
                    var bestPath = generateSimulatedPath([originLat, originLng], [destLat, destLng], 0.002);
                    var best = L.polyline(bestPath, {color: 'red', weight: 4}).addTo(map);
                    paths.push(best);
                    
                    // 更新路径信息
                    var distance = calculateDistance(bestPath).toFixed(2);
                    document.getElementById('path-details').innerHTML = `
                        <p>起点: (${originLat}, ${originLng})</p>
                        <p>终点: (${destLat}, ${destLng})</p>
                        <p>总距离: ${distance} km</p>
                        <p>路径类型: ${useSpeed ? '基于时间' : '基于距离'}</p>
                        <p>生成的Spark数: ${numSparks}</p>
                    `;
                }, 1000);
            }
            
            function generateSimulatedPath(start, end, deviation) {
                var path = [start];
                var numPoints = 5 + Math.floor(Math.random() * 5);
                
                for (var i = 1; i < numPoints; i++) {
                    var ratio = i / numPoints;
                    var lat = start[0] * (1 - ratio) + end[0] * ratio;
                    var lng = start[1] * (1 - ratio) + end[1] * ratio;
                    
                    // 添加随机偏移
                    lat += (Math.random() - 0.5) * deviation;
                    lng += (Math.random() - 0.5) * deviation;
                    
                    path.push([lat, lng]);
                }
                
                path.push(end);
                return path;
            }
            
            function calculateDistance(path) {
                var distance = 0;
                for (var i = 0; i < path.length - 1; i++) {
                    distance += getDistanceFromLatLonInKm(path[i][0], path[i][1], path[i+1][0], path[i+1][1]);
                }
                return distance;
            }
            
            function getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
                var R = 6371; // 地球半径（千米）
                var dLat = deg2rad(lat2 - lat1);
                var dLon = deg2rad(lon2 - lon1); 
                var a = 
                    Math.sin(dLat/2) * Math.sin(dLat/2) +
                    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
                    Math.sin(dLon/2) * Math.sin(dLon/2); 
                var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
                var d = R * c; // 距离（千米）
                return d;
            }
            
            function deg2rad(deg) {
                return deg * (Math.PI/180)
            }
        </script>
    </body>
    </html>
    """
    return html_content

# 主函数
def main():
    # 设置高德地图 API Key
    amap_key = "20eab95559788e3ad3f71cf404c60821"  # 需要替换为实际的API Key
    
    # 构建路网
    print("正在构建路网...")
    G = build_road_network("海淀区, 北京")
    print(f"路网构建完成，包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边")
    
    # 设置示例起点和终点坐标（经度，纬度）
    origin = (116.34899139404298, 39.95976154261228)  # 北京邮电大学西门
    # destination = (116.315681,39.990138)   # 北京大学东南门
    # destination = (116.332148,39.994847)  # 清华园
    destination = (116.30986976728312, 39.99037781712655)  
    # 查找最佳路径
    print("正在计算最佳路径...")
    best_path, updated_G, fireworks = find_best_path(G, origin, destination, amap_key, use_speed=True, num_sparks=5)
    
    if best_path:
        print("找到最佳路径！")
        
        # 可视化路径
        print("生成可视化结果...")
        fig = visualize_paths(updated_G, best_path, fireworks, origin, destination)
        plt.savefig("best_route.png")
        print("静态可视化结果已保存为 'best_route.png'")
        
        # 创建交互式地图
        m = create_interactive_map(updated_G, best_path, fireworks, origin, destination)
        m.save("interactive_map.html")
        print("交互式地图已保存为 'interactive_map.html'")
        
        # 创建路径规划器HTML
        html_content = create_route_planner_html()
        with open("route_planner.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("路径规划器已保存为 'route_planner.html'")
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