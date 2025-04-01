# D3VO-tensorflow
D3VO tensorflow implementation

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/D3VO-tensorflow/total.svg)]() 


<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/D3VO-tensorflow">
 <img src="https://img.shields.io/github/forks/chansoopark98/D3VO-tensorflow">
 <img src="https://img.shields.io/github/stars/chansoopark98/D3VO-tensorflow">
 <img src="https://img.shields.io/github/license/chansoopark98/D3VO-tensorflow">
 </p>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/C++-00599C.svg?&style=for-the-badge&logo=cplusplus&logoColor=white"/>
 <img src ="https://img.shields.io/badge/g2o-4B32C3.svg?&style=for-the-badge&logo=g2o&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Tensorflow-FF6F00.svg?&style=for-the-badge&logo=Tensorflow&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Pandas-150458.svg?&style=for-the-badge&logo=Pandas&logoColor=white"/>
 <br>
</p>



# Installation

## 1. Setup virtual environment
```
conda create -n vslam python=3.10
conda activate vslam

## 2. Setup g2opy
```
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
pip install setuptools==58.2.0
python setup.py install
cd ../
pip install -r requierments.txt
```

## 2. 


