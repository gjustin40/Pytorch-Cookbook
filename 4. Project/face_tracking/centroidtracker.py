from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0             # ID 숫자
        self.objects = OrderedDict()      # existing object 목록
        self.disappeared = OrderedDict()  # lost일 때 frame 개수
        
        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid):                   # centroid 추가
        self.objects[self.nextObjectID] = centroid  # centroid 등록
        self.disappeared[self.nextObjectID] = 0     # lost 일 때 frame 개수 초기화
        self.nextObjectID += 1                      # Object ID +1 추가
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects):
        if len(rects) == 0: # 만약 new object 탐지가 없으면
            for objectID in list(self.disappeared.keys()): # 모든 기존 object
                self.disappeared[objectID] += 1            # disppear +1 추가
                
                # 만약 임계 disppear 넘으면 해당 object ID 삭제
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister[objectID]
                    
            return self.objects
        
        # new object 탐지되면 개수만큼 dummy array 만들고 추가
        inputCentroids = np.zeros((len(rects), 2), dtype='int')
        for (i, (startX, startY, endX, endY)) in enumerate(rects): # 추가
            cX = int((startX + endX) / 2)
            cY = int((startY + endY) / 2)
            inputCentroids[i] = (cX, cY)   # new object의 중심점 모음집
            
        # centroid tracking 부분
        # 기존에 아무 object도 없었으면 new object 중심점 그대로 ID목록에 추가
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
                
        # 기존에 object 갱신하기
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # 거리 구하기
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # 이 부분이 너무 어려웠는데 굳이 이렇게 빙빙 돌려가면서 구하는 이유
            # 만약 old object에서 disappear가 되면 new object가 더 적어짐
            # 그래서 그냥 argmax로 찾아버리면 index가 겹침
            # 직접 행렬 그려서 보면 이해가 됨(disappear 발생한 거 가정하면)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            # 새로운 centroid로 update
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                    
                objectID = objectIDs[row] # old object의 ID 불러와서
                self.objects[objectID] = inputCentroids[col] # new centroid 업데이트
                self.disappeared[objectID] = 0 # disappear 0으로 초기화
                
                usedRows.add(row)
                usedCols.add(col)
                
            # old object에 없던 new object의 centroid   -----  1번 경우
            # 또는
            # disappear가 되어 없어져야 하는 old object의 cetroid ----- 2번 경우
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[0])).difference(usedCols)
            
            # 만약 2번의 경우
            # D.shape = (기존 n , 새로운 m)
            if D.shape[0] >= D.shape[1]: # 만약 없어지면 
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                        
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
                    
        return self.objects
            
                