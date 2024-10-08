# CSC8370ProjectDemo

⚡ If you can run "FirstLesson.py" on your PC, then congratulations, you have mastered the knowledge of this lesson.

⚡ If you have no idea how to Configure the operating environment to run this .py file,
following me to talk about how to run it in the server.

### connecting to server using VScode by SSH
* Open SSH config File, and enter the following up:

      Host CSc8370
  
        HostName inspire-gpu.cs.gsu.edu
  
        Port 2220-2224
  
        User gpu1

* Input password, then try to remote connecting

### We first enter the folder '/user1' if we are in the folder '/home'
    cd user1

### Creating a new environment for project model

* we have to input this command firstly, since the virtual environment is build by nvidia docker.(Any other cases, We dont need this step)
```
Source .bashrc
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

### Upload the code using git command [We can also choose file transfer(SSH) to upload the code ourselves.]

* Creating a new folder 'user1' to save ours code
```
mkdir dong
```
* Enter the folder 'dong'
```
cd dong
```
* Download the code using git command

