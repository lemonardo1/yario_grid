import numpy as np
import time

class Yolo_to_tensor():
def yolo_to_tensor_by_class(yolo_boxes, class_mapping, grid_size=(16, 15)):
    # yolo_boxes
    # 2차원array
    # [[x 좌표 1 , y 좌표 1, x 좌표 2, y 좌표 2, 정확도, 클래스]]
    # 클래스 그룹의 개수에 따라 텐서 생성
    num_classes = len(class_mapping)  # 그룹 개수는 class_mapping의 키 개수
    tensors = [np.zeros(grid_size, dtype=int) for _ in range(num_classes)]
    grid_h, grid_w = grid_size
    
    # YOLO 형식: (class, x_center, y_center, width, height)
    for box in yolo_boxes:
        cls, x_center, y_center, width, height = box
        
        # 클래스가 속한 그룹 찾기
        group = None
        for key, class_list in class_mapping.items():
            if cls in class_list:
                group = key
                break
        
        if group is None:
            continue  # 클래스가 어떤 그룹에도 속하지 않으면 무시
        
        # 물체의 중심 좌표를 그리드 좌표로 변환
        grid_x = int(x_center * grid_w)
        grid_y = int(y_center * grid_h)
        
        # 좌표가 그리드 범위를 벗어나지 않도록 클램핑
        grid_x = min(max(grid_x, 0), grid_w - 1)
        grid_y = min(max(grid_y, 0), grid_h - 1)
        
        # 해당 그룹의 텐서에서 물체의 위치에 1을 할당
        tensors[group][grid_y, grid_x] = 1

    return tensors



def read_yolo_boxes(file_path):
    yolo_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            # 각 줄을 읽고 공백으로 분할하여 float로 변환
            parts = line.strip().split()
            yolo_boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return yolo_boxes

file_path = 'yolo_boxes.txt'
yolo_boxes = read_yolo_boxes(file_path)
# # 예시 YOLO 형식 입력
# yolo_boxes = [
#     (0, 0.2, 0.3, 0.1, 0.1),  # class, x_center, y_center, width, height
#     (1, 0.8, 0.6, 0.2, 0.2),
#     (2, 0.5, 0.5, 0.15, 0.15)
# ]

# 함수 실행

class_mapping = {
    1: 0, 3: 0, 5: 0,       # 그룹 0
    2: 1, 4: 1, 7: 1,       # 그룹 1
    8: 2, 9: 2,             # 그룹 2
    10: 3, 11: 3, 12: 3     # 그룹 3
}

class_mapping = {
    0 : [0,1,2],
    1 : [3],
    2 : [4,5,6,7,9,10,13,14,15,16,18,19],
    3 : [8,9,10,11,12,13,16,17,19]
}

start = time.time()
try_num = 100
for i in range(try_num):
    tensors = yolo_to_tensor_by_class(yolo_boxes, class_mapping)
end = time.time()

print(f"total time: {(end-start)/try_num}")
# # 각 클래스의 텐서를 출력
# for i, tensor in enumerate(tensors):
#     print(f"Class {i} tensor:")
#     print(tensor)
#     print()
