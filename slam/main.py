import cv2
import os
import numpy as np
from MonoVO import MonoVO
import numpy as np
import cv2
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vo.utils.visualization import Visualizer

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
		self.visualizer = Visualizer(draw_plane=False, is_record=False, video_fps=30, video_name="visualization.mp4")
		self.flip_transform = np.diag([1, -1, -1, 1])

	def run(self):
		current_idx = 0
		current_pose = np.eye(4)

		while self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				break

			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))

			# return depth, uncertainty, self.mp.frames[-1].pose, a, b
			outputs = self.mono_vo.process_frame(frame, optimize=True)

			if outputs is not None:

				depth, sigma, rel_pose, a, b = outputs

				# OpenCV camera coordinate system to Pyvis camera coordinate system
				rel_pose = self.flip_transform @ rel_pose @ self.flip_transform
			
				# Update the current pose
				current_pose = current_pose @ rel_pose
				
				self.visualizer.world_pose = current_pose
				self.visualizer.draw_trajectory(self.visualizer.world_pose, color="red", line_width=2)


			current_idx += 1
			self.visualizer.render()
        
		self.cap.release()
		cv2.destroyAllWindows()
        
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