import tensorflow as tf
import numpy as np
import random
import os

def set_seed(seed=42):
    # 1. Python 기본 난수 생성기 시드 설정
    random.seed(seed)
    
    # 2. NumPy 난수 생성기 시드 설정
    np.random.seed(seed)
    
    # 3. TensorFlow 난수 생성기 시드 설정
    tf.random.set_seed(seed)
    
    # 4. 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # # 5. GPU 연산 결정론적으로 설정 (TF 2.8 이상)
    # try:
    #     tf.config.experimental.enable_op_determinism()
    #     print("Op determinism enabled")
    # except:
    #     print("Warning: 완전한 결정론적 동작을 위해 TensorFlow 2.8 이상을 사용하세요.")
    
    # print(f"모든 랜덤 시드가 {seed}로 설정되었습니다.")