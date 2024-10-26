import cv2
import glob
import os

def create_video_from_boxed_images(folder_path, output_video_path, fps=30):
    """Create a video from images ending with '_boxed.png' in the specified folder."""

    # _boxed.png 파일만 선택하고 파일명 정렬
    image_files = sorted(glob.glob(os.path.join(folder_path, "*_boxed.png")))
    if not image_files:
        print("No _boxed.png files found in the specified folder.")
        return
    
    # 첫 번째 이미지로부터 비디오의 해상도 설정
    frame = cv2.imread(image_files[0])
    if frame is None:
        print("Error reading the first image.")
        return
    height, width, layers = frame.shape
    size = (width, height)
    
    # 비디오 작성기 생성
    if not output_video_path.endswith('.mp4'):
        output_video_path += '.mp4'  # 확장자가 없으면 .mp4 추가
    output_path = os.path.abspath(output_video_path)
    print(f"Saving video to {output_path}")  # 경로 출력으로 문제 해결 확인
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    if not video_writer.isOpened():
        print("Error: Video writer could not be opened.")
        return
    
    # 모든 이미지를 비디오 프레임으로 추가
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Error reading image {image_file}. Skipping...")
            continue
        video_writer.write(frame)  # 프레임 추가
    
    video_writer.release()
    print(f"Video saved at: {output_path}")


def display_images_as_video(folder_path, fps=60):
    """Display images ending with '_boxed.png' in the specified folder as a video sequence."""
    
    # _boxed.png 파일만 선택하고 파일명 정렬
    image_files = sorted(glob.glob(os.path.join(folder_path, "*_boxed.png")))
    if not image_files:
        print("No _boxed.png files found in the specified folder.")
        return
    
    # 이미지 파일을 하나씩 읽어와서 표시
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load image {image_file}. Skipping...")
            continue
        cv2.imshow('Video', image)
        # FPS에 맞추어 이미지를 표시
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break
    
    cv2.destroyAllWindows()


# create_video_from_boxed_images('./game_data', './video')
display_images_as_video('./game_data')