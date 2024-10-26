import os
import shutil
import glob
import numpy as np

def split_data_set(data_dir):
    # 폴더 경로 정의
    images_dir = os.path.join(data_dir, 'images', 'all')
    labels_dir = os.path.join(data_dir, 'labels', 'all')
    
    # 훈련, 검증, 테스트 폴더 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'labels', split), exist_ok=True)
    
    # 모든 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(images_dir, '*.png'))
    # 파일 목록을 무작위로 섞기
    np.random.shuffle(image_files)
    
    # 파일 분할
    num_files = len(image_files)
    train_end = int(num_files * 0.7)
    val_end = train_end + int(num_files * 0.15)
    
    # 파일을 각 폴더로 복사
    for i, image_file in enumerate(image_files):
        base_name = os.path.basename(image_file)
        label_file = os.path.join(labels_dir, base_name.replace('.png', '.txt'))
        
        if i < train_end:
            split_type = 'train'
        elif i < val_end:
            split_type = 'val'
        else:
            split_type = 'test'
        
        # 이미지 파일 복사
        shutil.copy(image_file, os.path.join(data_dir, 'images', split_type, base_name))
        # 레이블 파일 복사
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(data_dir, 'labels', split_type, base_name.replace('.png', '.txt')))
def verify_data_split(data_dir):
    # 각 폴더의 이미지 및 레이블 파일 수 확인
    for split in ['train', 'val', 'test']:
        images_path = os.path.join(data_dir, 'images', split)
        labels_path = os.path.join(data_dir, 'labels', split)
        
        # 이미지와 레이블 파일 목록 얻기
        image_files = glob.glob(os.path.join(images_path, '*.png'))
        label_files = glob.glob(os.path.join(labels_path, '*.txt'))
        
        # 파일 수 출력
        print(f"{split.upper()} - Images: {len(image_files)}, Labels: {len(label_files)}")
        
        # 각 이미지에 대한 레이블 파일 존재 여부 확인
        missing_labels = 0
        for image_file in image_files:
            label_file = os.path.join(labels_path, os.path.basename(image_file).replace('.png', '.txt'))
            if not os.path.exists(label_file):
                missing_labels += 1
                print(f"Missing label for image: {image_file}")
        
        if missing_labels == 0:
            print(f"All images in {split} have corresponding labels.")
        else:
            print(f"There are {missing_labels} missing labels in {split}.")

# 사용 예시
verify_data_split('dataset')

# # 사용 예시
# split_data_set('dataset')
