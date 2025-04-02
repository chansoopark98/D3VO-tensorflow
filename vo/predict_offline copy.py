import cv2
import os
import numpy as np
from utils.visualization import Visualizer

DEBUG = True
PER_FRAME_ERROR = True

# from vo.utils.visualization import Visualizer

def offline_vo(cap, visualizer):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[427.0528736,   0.       , 328.9062192],
	   [  0.       , 427.0528736, 230.6455664],
	   [  0.       ,   0.       ,   1.       ]])
	
	poses = np.load('./output_pose.npy')
	print(poses.shape)
	
	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:				
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			
		else:
			break
		
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:
				break	
		visualizer.render()
	visualizer.close()

if __name__ == "__main__":
	video_path = '/media/park-ubuntu/park_cs/slam_data/mars_logger/test_2/2025_02_27_11_06_47/movie.mp4'
	cap = cv2.VideoCapture(video_path)
	W = 640
	H = 480
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	visualizer = Visualizer(draw_plane=False, is_record=False, video_fps=30, video_name="visualization.mp4")
	offline_vo(cap, visualizer)
    