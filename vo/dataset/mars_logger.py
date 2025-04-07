import os
import glob
import pandas as pd
import numpy as np
import cv2

class MarsLoggerHandler(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/mars_logger'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 1
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        
        self.train_data = self.generate_datasets(fold_dir='train', shuffle=True)
        self.valid_data = self.generate_datasets(fold_dir='valid', shuffle=False)
        self.test_dir = os.path.join(self.root_dir, 'test')
        self.test_data = self.generate_test(test_dir=self.test_dir)

    def _extract_video(self, scene_dir: str, camera_name, camera_data: pd.DataFrame) -> int:
        video_file = os.path.join(scene_dir, 'movie.mp4')
        rgb_save_path = os.path.join(scene_dir, 'rgb')

        # Ensure output directory exists
        if not os.path.exists(rgb_save_path):
            print(f'Extracting video file: {video_file}')
            os.makedirs(rgb_save_path, exist_ok=True)

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_file}")

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_name = os.path.join(rgb_save_path, f'rgb_{str(idx).zfill(6)}.jpg')
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
                cv2.imwrite(rgb_name, frame)
                idx += 1

            cap.release()
            cv2.destroyAllWindows()

        dataset = glob.glob(os.path.join(rgb_save_path, '*.jpg'))
        data_len = len(dataset)

        rgb_sample = cv2.imread(dataset[0])

        img_h, img_w, _ = rgb_sample.shape

        if camera_name == 'S22':
            original_image_size = (3000, 4000)
            fx = 2.66908046e+03
            fy = 2.67550677e+03
            cx = 2.05566387e+03
            cy = 1.44153479e+03
        else:
            original_image_size = (img_h, img_w)
            cx = img_w / 2
            cy = img_h / 2
        
            fx = camera_data['fx[px]'].values[0]
            fy = camera_data['fy[px]'].values[0]
            raise ValueError(f"Camera name {camera_name} not recognized. Please check the camera metadata.")

        current_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Rescale intrinsic matrix
        resized_intrinsic = self._rescale_intrinsic(current_intrinsic, self.image_size, original_image_size)

        # Count and return the number of saved frames
        return data_len, resized_intrinsic

    def _rescale_intrinsic(self, intrinsic: np.ndarray, target_size: tuple, current_size: tuple) -> np.ndarray:
        # New shape = self.image_size (H, W)
        fx = intrinsic[0, 0] * target_size[1] / current_size[1]
        fy = intrinsic[1, 1] * target_size[0] / current_size[0]
        cx = intrinsic[0, 2] * target_size[1] / current_size[1]
        cy = intrinsic[1, 2] * target_size[0] / current_size[0]
        intrinsic_rescaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic_rescaled

    def _process(self, scene_dir: str, camera_name: str, is_test: bool=False) -> list:
        # load camera metadata
        camera_file = os.path.join(scene_dir, 'movie_metadata.csv')
        camera_data = pd.read_csv(camera_file)

        # load video .mp4
        length, resized_intrinsic = self._extract_video(scene_dir, camera_name, camera_data)
    
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'rgb', '*.jpg')))
        
        if is_test:
            step = 1
        else:
            step = 2

        samples = []
        for t in range(self.num_source, length - self.num_source, step):
            for i in range(self.num_source):
                sample = {
                    'source_left': rgb_files[t - self.num_source + i], # str
                    'target_image': rgb_files[t], # str
                    'source_right': rgb_files[t + self.num_source - i], # str
                    'intrinsic': resized_intrinsic # np.ndarray (3, 3)
                }
                samples.append(sample)
        return samples
            
    def generate_datasets(self, fold_dir, shuffle=False, is_test=False):
        # path = os.path.join(self.root_dir, fold_dir)
        camera_types = glob.glob(os.path.join(self.root_dir, '*'))

        datasets = []

        for camera_type in camera_types:
            current_fold = os.path.join(camera_type, fold_dir)
            camera_name = os.path.basename(camera_type)

            scene_files = sorted(glob.glob(os.path.join(current_fold, '*')))
            
            for scene in scene_files:
                dataset = self._process(scene, camera_name, is_test)
                datasets.append(dataset)
            
        datasets = np.concatenate(datasets, axis=0)

        print('Current fold:', fold_dir)
        print(f'  -- Camera types: {camera_types}')
        print(f'  -- dataset size: {datasets.shape}')

        if shuffle:
            np.random.shuffle(datasets)
        return datasets
    
    def generate_test(self, test_dir):
        datasets = []

        scene_files = sorted(glob.glob(os.path.join(test_dir, '*')))
        
        for scene in scene_files:
            dataset = self._process(scene, 'S22')
            datasets.append(dataset)

        datasets = np.concatenate(datasets, axis=0)
        return datasets

if __name__ == '__main__':
    import yaml
    
    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = MarsLoggerHandler(config)
    data_len = dataset.train_data.shape[0]
    for idx in range(data_len):
        a = 1
        # print(dataset.train_data[idx])