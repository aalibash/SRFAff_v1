cd /******************************************************************************************
README for Affordance detection using Structured Random Forests (SRF) 
v1.1 - Added in code for Cornell grasping dataset, Mar 2015 
v1.0 - First public release, Feb 2015

If you use this code or the UMD RGB-D Part Affordance Dataset, please cite:
A. Myers, C.L. Teo, C. Fermüller and Y. Aloimonos, 
"Affordance Detection of Tool Parts from Geometric Features", Proc. IEEE
Int'l Conference on Robotics and Automation (ICRA), Seattle, WA, 2015

For more details and to obtain the RGB-D Affordance dataset, visit the project webpage:
structured forests
http://www.umiacs.umd.edu/research/POETICON/geometric_affordance/

This software was developed under 64-bit Linux with Matlab R2014a. 
There is no guarantee it will run on other operating systems or Matlab versions 
(though it probably will).

Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
Licensed under the Simplified BSD License [see license.txt]
*******************************************************************************************
Overview
--------
This code release demonstrates the Structured Random Forest (SRF) Affordance detector introduced in [1]. The package includes all source code for training the SRF for different target affordances and for performing fast inference given a test input image. Evaluation procedures are also provided to reproduce the results reported in [1]. 

OPTIMIZATION NOTE: The Matlab parallel toolbox is used in certain portions of the code. OpenMP is also used in some mex C++ calls, and will speed things up as well. 

Quick start
-----------
NOTE: if all you want is to run the detector using pre-trained models, just follow steps 5 and 6.

1. Download and unzip the contents of this package into one directory [SRFAff_v1/].

2. Start Matlab, make sure [SRFAff_v1/] and its subdirectories are within the search path.

3. Download and unzip the RGB-D Affordance dataset (see above) and copy the desired test/train data into [SRFAff_v1/data/Affordance_Part_Data/test/ or train/]. 

4. [optional] To train a SRF for a target affordance: run script_trainSRFAff.m. This step is optional if you use the precomputed models at [SRFAff_v1/models/forest/precomputed_models/]. See "Training an affordance SRF" section below for details.

5. cd into [SRFAff_v1/detection/private/] and run compile.m to compile affDetectMex function.

6. To test a trained SRF affordance model: run script_runDetectionDemo.m. If you use the precomputed models (see step 4), copy them into [SRFAff_v1/models/forest/]. See "Inference using an affordance SRF" section below for details.
 
Folder contents and structure
-----------------------------
The main folder [SRFAff_v1/] contains the following main script files:
a. script_trainSRFAff.m 	-- Trains an affordance SRF, see "Training an affordance SRF" section below for details.

b. script_runDetectionDemo.m	-- Detects a target affordance given the trained affordance SRF. See "Inference using an affordance SRF" section below for details.

c. script_evalAffordanceRes_cd.m -- Reproduces the F^{w}_{\beta} results per affordance category reported in table II of [1]. See "Evaluating the affordance SRF" section for details.

d. script_evalAffordanceResRanked_cd.m -- Reproduces the R^{w}_{\beta} results per affordance category reported in table II of [1]. See "Evaluating the affordance SRF" section for details.

e. script_trainSRFAff_cornell.m --  Trains an affordance SRF using the Cornell grasping dataset [2]. See [cornell/README_cornell.txt] for details.

f. script_evalAffordanceRes_cornell.m -- Evaluates the SRF performance using the recognition and detection metrics of [2]. See [cornell/README_cornell.txt] for details.
 
The subdirectories contain specific procedures and data: 

1. [cornell/] 		-- procedures for training and evaluating the SRF over the Cornell grasping dataset [2]. See README_cornell.txt for details.

2. [data/] 		-- location of dataset + precomputed features. See README_data.txt for details.

3. [detection/] 	-- procedures for detecting a target affordance using a trained SRF.

4. [evaluation/] 	-- procedures for reproducing the evaluation results using the F^{w}_{\beta} and R^{w}_{\beta} metrics reported in [1].

5. [features/]  	-- procedures for computing fast features (normals, curvatures and shape index) for training/testing the affordance SRF model.

6. [models/] 		-- saves the initial decision trees in [models/tree/] and final affordance SRF in [models/forest/].

7. [toolbox/] 		-- a subset of Piotr's Image & Video Toolbox with precompiled mex functions in [toolbox/private/]. See README_toolbox.txt for details. For documentation and information on using and recompiling the entire toolbox, go to http://vision.ucsd.edu/~pdollar/toolbox/doc/ 

8. [training/] 		-- procedures for training an affordance SRF.


Training an affordance SRF
--------------------------
The main script you need to use script_trainSRFAff.m. After following the steps in "Quick Start" and making sure that the dataset is in the correct location. Follow these steps to train the affordance SRF:

1. Set "bCleanDepth=1" or "bCleanDepth=0". This will tell the code to train using either precomputed features (1) or fast features (0). For the former, the model will be suffixed with '_AF3_3Dp_N_S1', which corresponds to results reported in the paper. For the latter, the model will be suffixed with '_AF3_3Dp_F'. 

2. Set "target=<affordance_label string>". Valid affordance target strings can be found in label_classes.m (except 'background'). 

3. Set "opts.useParfor=1" or "opts.useParfor=0". If (1), tells the code to train several decision trees in parallel. Do this if your setup has sufficient memory. For the RGB-D affordance dataset, each decision tree consumes ~12GB of ram. If not, you can choose (0) and train each tree sequentially.  Training each tree takes approximately 20 minutes for the RGB-D affordance dataset.

4. Set "opts.treeTrainID=1:opts.nTrees" to tell the training code to train from tree number 1 to opt.nTrees=8 (by default). Modify this if you want to train a smaller number of trees in parallel (due to memory limitations).

5. Run script_trainSRFAff.m

6. Once all trees are trained, they will be saved in [models/tree/]. To train the final affordance SRF, set "opts.treeTrainID = -1" and rerun script_trainSRFAff.m. The final affordance SRF model will be saved in [models/forest/].


Inference using an affordance SRF
---------------------------------
The main demo script you need is script_runDetectionDemo.m. After following the steps in "Quick Start", ensure that your trained SRF model is in [models/forest/] or alternatively, copy a precomputed model from [models/forest/precomputed_models/]. Follow these steps to test the trained affordance SRF model given a test RGB-D image:

1. Set "bCleanDepth=1" or "bCleanDepth=0". (1) Tells the code to look for an SRF model with suffix '_AF3_3Dp_N_S1' trained with precomputed features and (0) tells the code to look for an SRF model with suffix '_AF3_3Dp_F' trained with fast features.

2. Set "target=<affordance_label string>". Valid affordance target strings can be found in label_classes.m (except 'background'). 

3. Run script_runDetectionDemo.m. If everything works correctly, the input test RGB and detection results will be shown. You can change the RGB-D image used + features by modifying the variables "rgbFN, depthFN, dataFN" accordingly.

 
Evaluating the affordance SRF
-----------------------------
The main scripts are: 1) script_evalAffordanceRes_cd.m and 2) script_evalAffordanceResRanked_cd.m. 1) reproduces the results using the F^{w}_{\beta} metric and 2) reproduces the results using the  R^{w}_{\beta} metric reported in Table II of [1]. Note that running the evaluation over the RGB-D affordance dataset takes a long time (even with parallelization). Allocate ~10 hrs per evaluation script. 

To run script_evalAffordanceRes_cd.m, make sure you follow the steps in "Quick Start" and then follow these steps:
1. Ensure that the dataset + ground truth + precomputed features are in the correct locations as noted in "para.dir.*"

2. Make sure the trained SRF models (all seven) are in [models/forest/]. Note that the evaluation code works only on models trained with precomputed features, that is, suffixed with '_AF3_3Dp_N_S1'.

3. Run script_evalAffordanceRes_cd.m. Detection results will be saved in [data/Affordance_Part_Data/results/<SRF_model_name>_RES/test/]. The final Wfb scores are saved in "WFb_scores.mat" (positives) and "WFb_scores_neg.mat" (negatives). The script will read in these scores and provide the final average score reported in the paper [1].

To run script_evalAffordanceResRanked_cd.m, make sure you follow the steps in "Quick Start" and then follow these steps:
1. Ensure that the dataset + ground truth + precomputed features are in the correct locations as noted in "para.dir.*"

2. Make sure the trained SRF models (all seven) are in [models/forest/]. Note that the evaluation code works only on models trained with precomputed features, that is, suffixed with '_AF3_3Dp_N_S1'.

3. Run script_evalAffordanceResRanked_cd.m. Results will be saved in [data/Affordance_Part_Data/results/<SRF_model_name>_RES_ranked/test/]. Intermediate ranked Wfb scores are saved in "WFbRanked_scores_N_posTMP.mat" (positives) and "WFbRanked_scores_N_negTMP.mat" (negatives). The final ranked Wfb scores are saved in "WFbRanked_scores_N.mat". The script will read in both scores and provide the final average score reported in the paper [1].

References
----------
[1] A. Myers, C.L. Teo, C. Fermüller and Y. Aloimonos, "Affordance Detection of Tool Parts from Geometric Features", Proc. IEEE Int'l Conference on Robotics and Automation (ICRA), Seattle, WA, 2015

[2] Ian Lenz, Honglak Lee, Ashutosh Saxena, "Deep Learning for Detecting Robotic Grasps", To appear in International Journal of Robotics Research (IJRR), 2014.

*******************************************************************************************
Any questions or bugs please email me [cteo-at-cs.umd.edu].





