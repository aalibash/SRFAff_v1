% Evaluation script to load a trained SRF affordance model and 
% run it over the entire dataset -- highly parallelized
% Produces R^w_{\beta} scores reported in Table II of the paper.
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
%% set paths
para.dir.code = pwd;  
para.dir.data = 'data/Affordance_Part_Data/'; % location of dataset
para.dir.rank = 'data/label_rank/';  % ranked ground truth labels
% precomputed data
para.dir.cleanDepth='data/depth_clean/'; % (~13G)
para.dir.cleanNorm='data/normals/'; % (~38G)
para.dir.cleanCurvature='data/curvature/'; % (~9.3G)

addpath(genpath(pwd)); addpath(para.dir.data);

label_classes; 
%{'background', 'grasp', 'cut', 'scoop', 'contain', 'pound','support', 'wrap-grasp'};
tS=2; tE=8; % ignore 'background', so start from 2
ModelsC=cell(tE,1);
trainDataC=[]; testDataC=[];
modelVersion='_AF3_3Dp_N_S1';
%% load all models at once
for tID=tS:tE 
    % labels are zero-indexed so we minus 1
    target=affordance_label{tID};
    targetID=tID-1;

    modelFnm=['modelFinal_' target modelVersion];
    forestDir = fullfile(pwd,'models','forest');
    forestFn = fullfile(forestDir, modelFnm);
    if(exist([forestFn '.mat'], 'file'))
      load([forestFn '.mat']); 
    else
        error('model not found');
    end
    assert(targetID==model.opts.targetID);
    %% Set model parameters
    model.opts.multiscale=0;            % set to 0 to reproduce results in paper
    model.opts.nTreesEval=8;            % set to 8 
    model.opts.nThreads=8;              % max number threads for evaluation
    model.opts.dataDir=para.dir.data;
    model.opts.cleanDepthDir=para.dir.cleanDepth;
    model.opts.cleanNormDir=para.dir.cleanNorm;
    model.opts.cleanCurvatureDir=para.dir.cleanCurvature;
    model.opts.bCleanDepth=1;
    model.opts.cropD={(40:469),(20:589)};
    model.opts.rankDir=para.dir.rank;

    ModelsC{tID,1}=model;
end

%% Get train and test image filepaths
toolsIds = dlmread('tool_type.txt');
testGtDir = [model.opts.dataDir 'test/'];
[testData] = getDataFP_ranked_cd(ModelsC, testGtDir, toolsIds);
dirSaveRes=fullfile(para.dir.data, 'results',['modelFinal' modelVersion '_RES_ranked']);
if ~exist(dirSaveRes, 'dir'), mkdir(dirSaveRes); end;
TestResFN=fullfile(dirSaveRes, 'test', 'WFbRanked_scores_N.mat');

%% compute WFb for all affordances - takes a LONG time (~10 hrs) for the entire RGB-D Affordance dataset
fprintf('Extracting testing results: %d positives | %d negatives\n', testData.nImgsP, testData.nImgsN);
tic; extractAffordanceResFn_ranked_cd(ModelsC, testData, [dirSaveRes '/test/']); toc;

%% summarize results
wFbTest=dlmread(TestResFN); wFbTestMean=nanmean(wFbTest,2)
fprintf('Averaged test ranked WFb: %f\n',nanmean(wFbTestMean));

