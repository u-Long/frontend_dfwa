from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from ialgo import TravelingSalesman
from ialgo import AmapDriving
from ialgo import iCenterline

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名，生产环境应限制为具体前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

class iloc(BaseModel):
    lat: float
    lng: float


class Location(BaseModel):
    pts: List[iloc]


@app.post("/location/")
async def receive_location(location: Location):
    # 处理前端传入的坐标点
    pnts = [{"lat": _.lat, "lng": _.lng} for _ in location.pts]
    otravel = TravelingSalesman(pnts)
    pnts = otravel.find_shortest_path()  # 点排序
    path = []  # 高德规划路径 多段
    for fp, tp in zip(pnts, pnts[1:]):
        oAmap = AmapDriving('20eab95559788e3ad3f71cf404c60821')  # 这里为高德web服务API，需替换为有效的API-key
        print(fp, tp)
        org = '{},{}'.format(fp['lng'], fp['lat'])
        tpg = '{},{}'.format(tp['lng'], tp['lat'])    
        amap_path = oAmap.get_driving_route(org, tpg)
        path.append(amap_path)
    path = ';'.join(path)  # 高德路径拼接
    print(path)
    path = [eval(_) for _ in path.split(';')]
    print("=============================================")
    print(path)
    oline = iCenterline(path)  # 计算中心线 防止回头路
    res = oline.iSampleLine()
    # 格式同 Location
    coords = [(item['lng'], item['lat']) for item in res]

    return coords


@app.get("/")
async def iclick_html():
    return FileResponse('pathBaseMultiPoint.html')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)