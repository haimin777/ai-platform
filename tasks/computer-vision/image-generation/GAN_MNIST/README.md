Create and train simple GAN to create MNIST images
======
**GAN_MNIST** is a keras implementation of simple GAN for image generation with advanced logging via MLflow

#### Generation model results (random initialisation)
![Screenshot software](https://github.com/haimin777/ai-platform/blob/master/tasks/computer-vision/image-generation/GAN_MNIST/noise.png "screenshot software")

#### Generation model results (10 epochs)
![Screenshot software](https://github.com/haimin777/ai-platform/blob/master/tasks/computer-vision/image-generation/GAN_MNIST/10epochs.png "screenshot software")

#### Generation model results (50 epochs)
![Screenshot software](https://github.com/haimin777/ai-platform/blob/master/tasks/computer-vision/image-generation/GAN_MNIST/50epochs.png "screenshot software")

## Install Conda
!wget -c https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh

!chmod +x Anaconda3-2019.07-Linux-x86_64.sh
!bash ./Anaconda3-2019.07-Linux-x86_64.sh -b -f -p /usr/local

## run project

mlflow run tasks/computer-vision/image-generation/GAN_MNIST
## Contact
#### Developer

* e-mail: alexhaimin@gmail.com
* Telegram: @WasteML112