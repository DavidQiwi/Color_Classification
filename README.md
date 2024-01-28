<h1>Color Classification</h1>
Training and testing functions for predicting color of the blocks using CNN.

<h2>Installation</h2>
pip install torch

<h2>Dataset Preparation</h2>
Download the directories from <b>images</b> folder from GitLab https://git.rwth-aachen.de/justin.pratt/ki-demonstrator/-/tree/main/images?ref_type=heads,
where only the folders blue, red and yellow are needed. After that put them into project's directory folder <b>parts</b>.
The structure should look as following:<br>   
<b>parts</b><br> 
|--blue<br> 
|--red<br> 
|--yellow<br> 

<h2>Training</h2>
There are two common possibilities to run <b>train.py</b>. You can just run the script in the IDE or via command line 
firtly changed the directory and then typing <b>python train.py</b>.<br>
After that the model will be saved.<br>
<i>Default hyperparameters</i>:<br>
*BATCH_SIZE = 32<br>
*HIDDEN_UNITS = 10<br>
*LEARNING_RATE = 0.01<br>
*device = cpu (or cuda, if available)
