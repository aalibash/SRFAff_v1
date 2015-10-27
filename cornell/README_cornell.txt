******************************************************************************************
README for training and evaluating a grasping affordance SRF using the Cornell grasping dataset [1].

Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
Licensed under the Simplified BSD License [see license.txt]
*******************************************************************************************
This folder contains functions for training and evaluating the SRF over the Cornell grasp dataset [1]. The functions are as follows.

In the main folder:
1) script_trainSRFAff_cornell.m: trains a SRF using [1] to provide positive and negative rectangles for learning good graspable regions. See "Training the SRF" section for details.

2) script_evalAffordanceRes_cornell.m: uses the trained SRF and extract grasp affordance prediction over a testing subset in [1]. Evaluates the prediction using the recognition and detection rates proposed by [2].

In this folder [/cornell]:
1) srfAffTrain_cornell: procedures for training an affordance SRF using [1].
2) evaluateCornellRecognition_fn: computes recognition rates [r_a in table IIc of the paper]
3) evaluateCornellDetection_fn: computes detection rates [d_a in table IIc of the paper]
4) readGraspingPcd.m: utility function for reading in depth data from [1].
5) unpackRGBFloat.m: utility function for reading in depth data from [1].

See [1] and [2] for details on the dataset.

Quick start
-----------

1. Download and unzip the contents of this package into one directory [SRFAff_v1/].

2. Start Matlab, make sure [SRFAff_v1/] and its subdirectories are within the search path.

3. Download and unzip [1] and copy [/rawData/] into [data/DeepGraspingData/rawData/]/

4. To train a SRF: run script_trainSRFAff_cornell.m. See "Training the SRF" section below for details. A precomputed model [models/forest/precomputed_models/modelFinal_grasp_AF_3Dp_C.mat] with its test split [models/forest/precomputed_models/modelFinal_grasp_AF_3Dp_C_testDat.mat] is available.

5. cd into [SRFAff_v1/detection/private/] and run compile.m to compile affDetectMex function.

6. To evaluate the trained SRF model: run script_evalAffordanceRes_cornell.m. If you use the precomputed model (see step 4), copy the model into [SRFAff_v1/models/forest/]. Also, copy the test split data (step 4) into [SRFAff_v1/models/]. See "Evaluation" section below for details.

Training the SRF
----------------
The main script you need to use is script_trainSRFAff_cornell.m. After following the steps in "Quick Start", make sure that [1] is in the correct location (step 3 in Quick Start). Follow these steps to train the SRF:

1. Set "opts.useParfor=1" or "opts.useParfor=0". If (1), tells the code to train several decision trees in parallel. Do this if your setup has sufficient memory. For the RGB-D affordance dataset, each decision tree consumes ~10GB of ram. If not, you can choose (0) and train each tree sequentially.  Training each tree takes approximately 20 minutes using [1].

2. Set "opts.treeTrainID=1:opts.nTrees" to tell the training code to train from tree number 1 to opt.nTrees=8 (by default). Modify this if you want to train a smaller number of trees in parallel (due to memory limitations).

3. Run script_trainSRFAff_cornell.m. The code will randomly create a train/test split (following the online code of [1]) and saves the current test split into [models/modelFinal_grasp_AF_3Dp_C_testDat.mat] .

4. Once all trees are trained, they will be saved in [models/tree/]. To train the final affordance SRF, set "opts.treeTrainID = -1" and rerun script_trainSRFAff_cornell.m. The final affordance SRF model will be saved as [models/forest/modelFinal_grasp_AF_3Dp_C.mat].

Evaluation
----------
The main script you need to use is script_evalAffordancesRes_cornell.m.

1. Make sure the SRF model [models/forest/modelFinal_grasp_AF_3Dp_C.mat] exists.

2. Make sure that the test split data [SRFAff_v1/models/modelFinal_grasp_AF_3Dp_C_testDat.mat] exists. 

3. Ensure the variable "dataDir" is correctly pointing to [/rawData/] from [1].

4. Run script_evalAffordancesRes_cornell.m which will print out the recognition and detection rates for the current test split on the terminal screen.


References
----------
[1] http://pr.cs.cornell.edu/grasping/rect_data/data.php
[2] Ian Lenz, Honglak Lee, Ashutosh Saxena, "Deep Learning for Detecting Robotic Grasps", To appear in International Journal of Robotics Research (IJRR), 2014.

*******************************************************************************************
Any questions or bugs please email me [cteo-at-cs.umd.edu].










