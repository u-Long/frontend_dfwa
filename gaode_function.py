import requests
import json


# 获取指定区域的实时路况
def get_traffic_info(rectangle, api_key):
    url = "https://restapi.amap.com/v3/traffic/status/rectangle"
    params = {
        'key': api_key,
        'rectangle': rectangle,  # 格式：左下经度,左下纬度;右上经度,右上纬度
        'level': 6,  # 必填参数：道路等级（1-高速, 6-所有道路）
        'extensions': 'all'  # 详细路况
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        traffic_data = response.json()
        print("Traffic Data: ", traffic_data)
        # save traffic data to file
        with open('traffic_data.json', 'w') as f:
            json.dump(traffic_data, f, indent=4)
        if traffic_data['status'] == '1':
            return traffic_data['trafficinfo']
        else:
            print("API Error: ", traffic_data['info'])
    else:
        print("HTTP Error: ", response.status_code)
    return None

# 示例使用
rectangle = "116.351147,39.966309;116.357446,39.968066"
traffic_info = get_traffic_info(rectangle, '20eab95559788e3ad3f71cf404c60821')
if traffic_info:
    print("当前路况: ", traffic_info['description'])


