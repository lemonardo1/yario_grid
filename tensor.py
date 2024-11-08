import torch
class Tensor():
    def __init__(self, base_frame_count = 4):
        self.base_frame_count = base_frame_count
        self.mario_state_num = 3
        self.num_classes = 4 
        self.x_grid_num = 16
        self.y_grid_num = 15
        self.grid_size = self.x_grid_num * self.y_grid_num


        self.last_frame = 0
        # 외부에서 바로 접근 못하게 private로 
        self.__all_tensors = torch.zeros((self.base_frame_count, self.mario_state_num + self.num_classes * self.grid_size), dtype=torch.float)


    def update(self, mario_state, grid_x, grid_y, group_id, frame_num):
        
        frame_tensor = torch.zeros((self.mario_state_num + self.num_classes * self.grid_size,), dtype=torch.float)
        # 마리오만 업데이트하는 함수를 만들면 좀 더 성능상 이점이 있지만 종속성이 생길 우려가 있어 그냥 이렇게 처리함
        if mario_state != None:
            mario_state = min(int(mario_state), 2)
            frame_tensor[mario_state] = 1  # 마리오 상태 저장

        grid_1d_index = grid_x + grid_y * self.x_grid_num
        index = self.mario_state_num + group_id * self.grid_size + grid_1d_index

        frame_tensor[index] = 1

        # frame_num 0 ~ base_frame_count - 1 의 범위
        self.__all_tensors[frame_num] = frame_tensor
        self.last_frame = frame_num

    def get_tensor(self):
        if self.last_frame == self.base_frame_count - 1:
            final_tensor = self.__all_tensors.clone()
            final_tensor = final_tensor.view(-1)
            self.__all_tensors.fill_(0)
            return final_tensor
        return None
    
    def get_base_frame_count(self):
        return self.base_frame_count