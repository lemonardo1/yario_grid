class ClassMapping():
    def __init__(self):
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

        self.rgb_mapping = {
            0: (100,100,100),
            1: (255,0,0),
            2: (0,255,0),
            3: (0,0,255),
        }


    def get_group_id(self, class_id):
        group_id = None
        for key, class_list in self.class_mapping.items():
            if class_id in class_list:
                group_id = key
                break
        
        return group_id
    
    def get_color_by_group_id(self, group_id):
        return self.rgb_mapping[group_id]

    # def get_color_by_class_id(self, class_id):
    #     group_id = self.get_group_id(class_id)
    #     return self.rgb_mapping[group_id]
    
        