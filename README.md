# ExMeshCNN
ExMeshCNN: An Explainable Convolutional Neural Network Architecture for 3D Shape Analysis

Accepted in ACM SIGKDD 2022 (https://kdd.org/kdd2022/)
# Requirements
- ubuntu 20.04.3 LTS

- CUDA == 11.1
- cudnn == 8.1.0
- numpy == 1.19.5
- matplotlib == 3.4.3 
- openmesh == 1.1.3
- pymeshlab == 2021.7
- torch == 1.7.1+cu110
- torchvision == 0.8.2+cu110
- trimesh == 3.9.29

# Dataset
- SHREC (https://github.com/ranahanocka/MeshCNN)
- Cubes (https://github.com/ranahanocka/MeshCNN)
- Manifold40 (https://github.com/lzhengning/SubdivNet)
- ModelNet40 (https://modelnet.cs.princeton.edu/)

- Download: https://drive.google.com/file/d/19Typ-Yt1oBDwr9q5uCawa6EHqsJMH-Vn/view?usp=sharing
  - These datasets were created to have an equal number of faces and follow a manifold format.

# Usage
- Preprocessing: python resize_manifold.py && python make_dataset.py
- Train the model: python trainer.py
- Explain: python Grad-CAM.py

code description

- resize_manifold.py
  - Make it follow the manifold format and have the same number of faces.
- make_dataset.py
  - It transforms into the data structure suggested in the paper to efficiently conduct learning and testing.
- trainer.py
  - Train the model.
- Grad-CAM.py
  - Describe the model using the GradCAM method.

Hyperparameters are defined in the code.

For proper testing, hyperparameters of each file must be modified.
