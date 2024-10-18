# CSC8370ProjectDemo_24fall

⚡ If you can run "FirstLesson.py" on your PC, then congratulations, you have mastered the knowledge of this lesson.

⚡ If you have no idea how to Configure the operating environment to run this .py file,
following me to talk about how to run it in those servers provided by Professor Wang.

⚡ Before you leave, please show me that you can successfully run the [FirstLesson.py](https://github.com/DongYang26/CSC8370ProjectDemo/blob/main/FirstLesson.py) file in your environment.

⚡ If you would like to use the server in this lesson, please register now in the [form](https://docs.google.com/spreadsheets/d/1WYS-VxcrPQoU8dC_G4ycknN3thgoKp5cSDy71u_1vGw/edit?usp=sharing).
### 1. Connecting to server using VScode by SSH
* Open SSH config File in Vscode, and enter the following up:

      Host CSc8370
  
        HostName inspire-gpu.cs.gsu.edu
  
        Port 2220   # We have 5 ports (2220-2224), each port corresponds to a virtual server.
  
        User gpu1

* Input password, then try to remote connecting

### 2. We first enter the folder '/gpu1' if we are in the folder '/home'
    cd gpu1

### 3. Creating a new environment for project model

❗️Assuming you have already installed conda, if not, please refer to this [document](https://github.com/DongYang26/CSC8370ProjectDemo/blob/main/Preparation.md). Those servers have installed conda.
* we have to input this command firstly, since the virtual environment is build by nvidia docker.(Any other cases, We dont need this step)
```
Source .bashrc
```
or using conda init command
```
conda init
```
* Using **conda** command to check the environment we have built.
```
conda env list
```
* Using **conda** command to build a new environment for our project.
```
conda create -n your_env_name python=X.X.X
```
for example: conda create -n dong_test python=3.8.19
* Activating the new environment by using **conda** command (Important🌟🌟🌟)
```
conda activate dong_test
```

### 4. Install the pytorch according to the pytorch Official Website
* checking the CUDA version
```
nvidia-smi
```
* Go to the [pytorch Official Website](https://pytorch.org/get-started/locally/), copy the command according to CUDA version
for example:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 5. Download the code using git command [We can also choose file transfer(SSH) to upload the code ourselves.]

* Creating a new folder 'dongProject' to save ours code
```
mkdir dongProject
```
* Enter the folder 'dongProject'
```
cd dongProject
```
* Download the code into folder dongProject using git command
```
git clone https://github.com/DongYang26/CSC8370ProjectDemo.git
```
* We have to iteratively enter the code folder until we can see file 'FirstLesson.py' using 'ls'command (Important🌟🌟🌟)
```
cd CSC8370ProjectDemo
```
* Try to run the code.
```
python FirstLesson.py
```
* we need to install it if required package is not in the environment.
```
conda install package_name_missed
```

