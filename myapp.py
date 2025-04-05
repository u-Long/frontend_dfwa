from flask import Flask, request, jsonify
from flask_cors import CORS 
import networkx as nx
import osmnx as ox
from backend import build_road_network, find_best_path, create_interactive_map  # 导入你的方法

app = Flask(__name__)
CORS(app) 

# 路网构建
G, diG = build_road_network("海淀区, 北京", simplify=True)

@app.route('/plan_route', methods=['POST'])
def plan_route():
    data = request.json
    origin = (float(data['origin_lng']), float(data['origin_lat']))
    destination = (float(data['dest_lng']), float(data['dest_lat']))
    amap_key = data['amap_key']
    use_speed = data['use_speed']
    num_sparks = int(data['num_sparks'])
    # 打印参数
    print(f"Origin: {origin}, Destination: {destination}, Amap Key: {amap_key}, Use Speed: {use_speed}, Num Sparks: {num_sparks}")

    # 计算最佳路径
    best_path, updated_G, fireworks, all_paths, road_info = find_best_path(G, diG, origin, destination, amap_key, use_speed, num_sparks, really_use_api=True)

    if best_path:
        # 可视化路径并保存地图
        m = create_interactive_map(updated_G, best_path, all_paths, origin, destination)
        m.save("interactive_map.html")

        coordinates = [(G.nodes[node].get('y', 'Unknown'), G.nodes[node].get('x', 'Unknown')) for node in best_path]
        print(coordinates)



        return jsonify({
            "best_path": coordinates,
            "road_info": road_info,
            # "fireworks": fireworks,
            # "all_paths": all_paths,
            "map_url": "interactive_map.html"
        })
    else:
        return jsonify({"error": "无法找到最佳路径"}), 400

if __name__ == '__main__':
    app.run()