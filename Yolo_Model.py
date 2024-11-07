import numpy as np
import time
import torch

from PIL import Image
from ultralytics import YOLO



class Yolo_Model():
    def __init__(self, x_pixel_num, y_pixel_num):
        # 들어오는 이미지의 픽셀 수
        self.x_pixel_num = x_pixel_num
        self.y_pixel_num = y_pixel_num

        # 클래스의 종류 (마리오, 아이템, 바닥, 적)
        self.num_classes = 4

        self.x_grid_num = 16
        self.y_grid_num = 15

        self.x_unit_length = self.x_pixel_num / self.x_grid_num
        self.y_unit_length = self.y_pixel_num / self.y_grid_num

        self.model = YOLO("best.pt")
        self.frame_count = 0
        self.frame_base = 4 # 이 프레임마다 텐서를 반환함

        self.grid_size = self.x_grid_num * self.y_grid_num
        # 마리오 상태 포함 텐서 정의 (4, 1 + num_classes * grid_size)
        # self.all_tensors = np.zeros((4, 1 + self.num_classes * self.grid_size), dtype=int)
        # PyTorch 텐서로 초기화
        self.mario_state_num = 3
        self.all_tensors = torch.zeros((4, self.mario_state_num + self.num_classes * self.grid_size), dtype=torch.float)
        self.is_logging = False
        
        # self.class_mapping
        # 0: 마리오
        # 1: 적
        # 2: 아이템, 코인블록, 토관, 깃발, 깰 수 있는 블록
        # 3: 밟을 수 있는 블록

        self.class_mapping = {
            0 : [0,1,2],
            1 : [3],
            2 : [4,5,6,7,9,10,13,14,15,16,18,19],
            3 : [8,9,10,11,12,13,16,17,19]
        }

    def get_class_mapping_id(self, cls):
        group_id = None
        for key, class_list in self.class_mapping.items():
            if cls in class_list:
                group_id = key
                break
        
        return group_id

    def set_logging(self):
        self.is_logging = True

    def unable_logging(self):
        self.is_logging = False

    def get_tensor(self, img: Image, mario_state: int):
        self.frame_count += 1

        with torch.no_grad():  # 연산을 추적하지 않도록 설정
            results = self.model(img, verbose=self.is_logging)

        frame_tensor = torch.zeros((self.mario_state_num + self.num_classes * self.grid_size,), dtype=torch.float)
        # frame_tensor = np.zeros((1 + self.num_classes * self.grid_size,), dtype=int)
        if mario_state > 2: # 2보다 큰 경우는 없는거로 취급 0: 작은 마리오, 1: 큰 마리오, 2: 불 마리오
            mario_state = 2
        frame_tensor[mario_state] = 1  # 마리오 상태 저장


        for result in results:
            for box in result.boxes:
                x1 = box.xyxy[:,0]
                x2 = box.xyxy[:,2]
                y1 = box.xyxy[:,1]
                y2 = box.xyxy[:,3]
                cls = box.cls

                group_id = self.get_class_mapping_id(int(cls))
                
                # 박스의 너비와 높이 계산
                widths = x2 - x1
                heights = y2 - y1

                # 그리드 크기로 변환
                x_grid_sizes = torch.round(widths / self.x_unit_length)
                y_grid_sizes = torch.round(heights / self.y_unit_length)


                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                grid_x = min(max(int(x_center / self.x_unit_length), 0), self.x_grid_num - 1) # 0 ~ self.x_grid_num - 1 사이의 값
                grid_y = min(max(int(y_center / self.y_unit_length), 0), self.y_grid_num - 1) # 0 ~ self.y_grid_num - 1 사이의 값
                grid_1d_index = grid_x + grid_y * self.x_grid_num


                index = self.mario_state_num + group_id * self.grid_size + grid_1d_index
                frame_tensor[index] = 1  # 해당 위치에 1을 채워넣음

                # y 크기가 2인 마리오이면 해당 칸보다 한칸 더 위도 채움
                if y_grid_sizes == 2:
                    index = index + max(grid_1d_index - 1, 0) * self.x_grid_num
                    frame_tensor[index] = 1
            
        self.all_tensors[self.frame_count] = frame_tensor
        self.frame_count += 1

        # self.frame_base 개의 프레임을 입력받은 후에 1개의 action을 취함
        if self.frame_count == self.frame_base:
            # start = time.time()/
            final_tensor = self.all_tensors.clone()
            final_tensor = final_tensor.view(-1)  # 모든 요소를 1차원으로 변환
            self.all_tensors.fill_(0)
            # self.all_tensors = np.zeros((4, 1 + self.num_classes * self.grid_size), dtype=int) # 초기화
            self.frame_count = 0
            # end = time.time()
            # print(f"tensor transform time: {end-start}")
            return final_tensor

        return None


    ########################## 이 아래는 테스트용 코드 ###############################
    def yolo_test(self, img_path):
        results = self.model(img_path)
        return results

    def yolo_to_tensor_by_class(self, yolo_boxes, class_mapping, grid_size=(16, 15)):
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



    def read_yolo_boxes(self, file_path):
        yolo_boxes = []
        with open(file_path, 'r') as file:
            for line in file:
                # 각 줄을 읽고 공백으로 분할하여 float로 변환
                parts = line.strip().split()
                yolo_boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        return yolo_boxes
    
    def test_yolo_output_txt(self, file_path):
        yolo_boxes = self.read_yolo_boxes(file_path)
        tensors = self.yolo_to_tensor_by_class(yolo_boxes, self.class_mapping)
        return tensors

