% Script for training SRF Affordance model with the Cornell grasping
% dataset.
% 
% If you use this code or the UMD RGB-D Part Affordance Dataset, please
% cite:
% A. Myers, C.L. Teo, C. Fermüller and Y. Aloimonos, 
% "Affordance Detection of Tool Parts from Geometric Features", Proc. IEEE
% Int'l Conference on Robotics and Automation (ICRA), Seattle, WA, 2015
%
% For more details and to obtain the RGB-D dataset, visit the project webpage:
% http://www.umiacs.umd.edu/research/POETICON/geometric_affordance/
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;
%% setup -- change this to suit your current setup
para.dir.code = pwd;
para.dir.data = 'data/DeepGraspingData/rawData/'; % location of Cornell grasping dataset
addpath(genpath(pwd)); addpath(para.dir.data);

%% set opts for training
target='grasp';                     % only grasp
opts=srfAffTrain_cornell();         % default options (good settings)
opts.modelDir='models/';            % model will be in models/forest
opts.modelFnm=['modelFinal_' target '_AF_3Dp_C'];       % model name
opts.dataDir=para.dir.data;

opts.nPos=5e5; opts.nNeg=5e5;       % decrease to speedup training and use less memory
opts.maxDepth=64;                   % max tree depth
opts.useParfor=1;                   % parallelize if sufficient memory (~10GB/Tree)
opts.nClasses = 2;                  % affordance vs. background
opts.nTrees = 8;                    % number of decision trees

% [1:8]: treeID to train in cluster, -1 to combine trees to train final SRF (see README.txt)
opts.treeTrainID = -1;              %1:opts.nTrees; % -1 


opts.CVpctile=50; % percentile to threshold Curvedness smoothing

% skip how many images? (for speed in training/testing)
opts.posSkip = 1;                   % [1] set to 1 to train all positives, else a positive number to get a subset
opts.negSkip = 1;                   % [1] set to 1 by default to train all negatives


% Ablation types
% 0: use 2D (RGB) features only, 
% 1: use Depth features (depth+gradient+gradient mag) only,
% 2: use 3D features only (Depth + normals + curvatures + shape index) - best (reported in paper)
opts.rgbd=2;        % DO NOT change unless you know what you are doing

%% train SRF affordance detector
if opts.treeTrainID > 0
    fprintf('Training decision tree AF: %s\n', target); 
else 
    fprintf('Training SRF forest AF: %s\n', target); 
end
tic, model=srfAffTrain_cornell(opts); toc;



