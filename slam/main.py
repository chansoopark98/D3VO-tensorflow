import cv2
import os
import numpy as np
from MonoVO import MonoVO
import numpy as np
import open3d as o3d
import cv2
import time

DEBUG = True
PER_FRAME_ERROR = True

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
		self.mono_vo = MonoVO(self.intrinsic)

		# Flip transform: OpenCV -> Open3D
		self.flip_transform = np.diag([1, -1, -1, 1])

		# Open3D 시각화 준비
		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window(window_name="Camera Trajectory", width=960, height=720)
		# 궤적 선의 두께 조정을 위해 render 옵션 설정
		render_opt = self.vis.get_render_option()
		render_opt.line_width = 5.0

		self.trajectory = o3d.geometry.LineSet()
		self.trajectory.points = o3d.utility.Vector3dVector([])
		self.trajectory.lines = o3d.utility.Vector2iVector([])
		self.trajectory.colors = o3d.utility.Vector3dVector([])
		self.vis.add_geometry(self.trajectory)

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

			# return depth, uncertainty, self.mp.frames[-1].pose, a, b
			outputs = self.mono_vo.process_frame(frame, optimize=True)

			if outputs is not None:
				depth, sigma, current_pose, a, b = outputs

				# 현재 카메라 포즈 불러오기 및 변환
				# current_pose = self.camera_poses[current_frame]
				current_pose = self.flip_transform @ current_pose @ self.flip_transform

				# 카메라 위치 추출 (translation)
				cam_position = current_pose[:3, 3]
				traj_points.append(cam_position)

				# 궤적 선 연결
				if current_frame > 0:
					traj_lines.append([current_frame - 1, current_frame])

				# LineSet 업데이트
				self.trajectory.points = o3d.utility.Vector3dVector(traj_points)
				self.trajectory.lines = o3d.utility.Vector2iVector(traj_lines)
				self.trajectory.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(traj_lines))

				self.vis.update_geometry(self.trajectory)

				# 현재 카메라 pose 좌표축 추가
				coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.cam_size)
				coord.transform(current_pose)
				self.vis.add_geometry(coord)
				self.coord_frames.append(coord)

				self.vis.poll_events()
				self.vis.update_renderer()
				time.sleep(0.01)

			current_frame += 1
        
		self.cap.release()
		cv2.destroyAllWindows()
		self.vis.run()
		self.vis.destroy_window()
        
		save_path = os.path.join('./output_pose.npy')
		np.save(save_path, self.mono_vo.mp.relative_to_global())
		print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	video_path = '/media/park-ubuntu/park_cs/slam_data/mars_logger/test_2/2025_02_27_11_06_47/movie.mp4'
	cap = cv2.VideoCapture(video_path)
	W = 640
	H = 480
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	OfflineRunner(video_path=video_path,
               camera_poses=None,
               intrinsic=np.array([[427.0528736,   0., 328.9062192],
                                   [0., 427.0528736, 230.6455664],
                                   [0.,   0.,   1.]]),
               image_size=(H, W)).run()
    #  (video_path=video_path,
    #            camera_poses=None,
    #            intrinsic=np.array([[427.0528736,   0., 328.9062192],
    #                                [0., 427.0528736, 230.6455664],
	# 							   [0.       ,   0.       ,   1.       ]])