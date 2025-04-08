import os
import glob
import pandas as pd
import numpy as np
import open3d as o3d
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vo.utils.visualization import Visualizer
import cv2
import time

class OfflineRunner:
    def __init__(self,
                 video_path: str,
                 camera_poses: np.ndarray,
                 intrinsic: np.ndarray,
                 image_size: tuple = (480, 640)):
        self.video_path = video_path
        self.image_size = image_size
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.camera_poses = camera_poses
        self.intrinsic = intrinsic
        
        # Flip transform: OpenCV -> Open3D
        self.flip_transform = np.diag([1, -1, -1, 1])
        self.visualizer = Visualizer(draw_plane=False, is_record=False, video_fps=30, video_name="visualization.mp4")

        self.cam_size = 0.1  # 시각화용 카메라 크기
        self.coord_frames = []

    def run(self):
        current_frame = 0
        traj_points = []
        traj_lines = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))

            # 현재 카메라 포즈 불러오기 및 변환
            current_pose = self.camera_poses[current_frame]
            # current_pose = self.flip_transform @ current_pose @ self.flip_transform

            self.visualizer.world_pose = current_pose
            self.visualizer.draw_trajectory(self.visualizer.world_pose)

            # 카메라 위치 추출 (translation)
            cam_position = current_pose[:3, 3]
            traj_points.append(cam_position)

            # 궤적 선 연결
            if current_frame > 0:
                traj_lines.append([current_frame - 1, current_frame])

        

            current_frame += 1
            if current_frame >= min(len(self.camera_poses), self.frame_count):
                break
            
            self.visualizer.render()
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    
    video_path = '/media/park-ubuntu/park_cs/slam_data/mars_logger/test/2025_01_06_18_21_32/movie.mp4'
    camera_poses = np.load('./output_pose.npy')  # (N, 4, 4) 형태의 카메라 포즈 배열
    intrinsic = np.array([[427.0528736, 0, 328.9062192],
                          [0, 427.0528736, 230.6455664],
                          [0, 0, 1]])

    runner = OfflineRunner(video_path, camera_poses, intrinsic)
    runner.run()