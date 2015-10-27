% Script for training SRF Affordance model
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

function [] = script_trainSRFAff(affordanceName)
    clearvars -except affordanceName; close all;
    %% setup -- change this to suit your current setup
    bCleanDepth=0; % 1- use precomputed depth/normals/curvature, else 0
    para.dir.code = pwd; % location of source code
    para.dir.data = 'data/Affordance_Part_Data/'; % location of UMD Affordance Parts dataset (get from webpage)
    % precomputed data (ignored if bCleanDepth=0)
    para.dir.cleanDepth='data/depth_clean/'; % (~13G)
    para.dir.cleanNorm='data/normals/'; % (~38G)
    para.dir.cleanCurvature='data/curvature/'; % (~9.3G)

    addpath(genpath(pwd)); addpath(para.dir.data);

    % load affordance_label={'background','contain','cut','grasp','pound','scoop','support','wrap-grasp'};
    label_classes; 
    %% Setup training parameters of SRF for a target affordance
    % Valid affordances are: 'contain','cut','grasp','pound','scoop','support','wrap-grasp'
    target=affordanceName; %'grasp'; 
    targetID=find(strcmp(affordance_label,target)==1)-1; % labels are zero-indexed so we minus 1
    if isempty(targetID), error('Invalid target affordance string, please recheck'); end;
    opts=srfAffTrain();                 % get default options
    opts.modelDir='models/';            % model will be in models/forest
    if bCleanDepth
        modelType='_AF3_3Dp_N_S1';      % for results reported in paper
    else
        modelType='_AF_3Dp_F';          % fast, real-time feature extraction (less accurate)
    end

    opts.modelFnm=['modelFinal_' target modelType];   % model name
    opts.nPos=5e4; opts.nNeg=5e4;       % decrease to speedup training and use less memory
    opts.maxDepth=64;                   % max tree depth
    opts.useParfor=0;                   % parallelize if sufficient memory (~12GB/Tree)
    opts.dataDir=para.dir.data;
    opts.cleanDepthDir=para.dir.cleanDepth;
    opts.cleanNormDir=para.dir.cleanNorm;
    opts.cleanCurvatureDir=para.dir.cleanCurvature;
    opts.nClasses = 2;                  % affordance vs. background
    opts.targetID = targetID;
    opts.nTrees = 8;                    % number of decision trees

    % [1:8]: treeID to train in cluster, -1 to combine trees to train final SRF (see README.txt)
    opts.treeTrainID = 1:opts.nTrees;              %1:opts.nTrees; % -1 

    opts.bCleanDepth=bCleanDepth;
    opts.cropD={(40:469),(20:589)};     % crop values of precomputed RGB-D affordance features
    opts.CVpctile=50;                   % percentile to threshold Curvedness smoothing from precomputed features

    % skip how many images? (for speed in training/testing)
    opts.posSkip = 3;                   % [3] set to 1 to train all positives, else a positive number to get a subset
    opts.negSkip = 1;                   % set to 1 by default to train all negatives

    % Ablation types
    % 0: use 2D (RGB) features only, 
    % 1: use Depth features (depth+gradient+gradient mag) only,
    % 2: use 3D features only (Depth + normals + curvatures + shape index) - best (reported in paper)
    opts.rgbd=2;                        % DO NOT change unless you know what you are doing

    %% train SRF affordance detector
    if opts.treeTrainID > 0
        fprintf('Training decision tree AF: %s\n', target); 
    else 
        fprintf('Training SRF forest AF: %s\n', target); 
    end
    tic, model=srfAffTrain(opts); toc;
    
end
