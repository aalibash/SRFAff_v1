% Script to apply pre-trained 'grasp' SRF affordance model (see
% script_trainSRFAff_cornell.m) and evaluate over the Cornell grasping dataset
%
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
dataDir = 'data/DeepGraspingData/rawData/'; % location of Cornell grasping dataset
addpath(genpath(pwd)); addpath(dataDir);

%% Setup data directories
target='grasp';
modelFnm=['modelFinal_' target '_AF_3Dp_C'];
dirSaveRes=fullfile(dataDir, 'results', [modelFnm '_RES']);
if ~exist(dirSaveRes, 'dir'), mkdir(dirSaveRes); end;
forestDir = fullfile(pwd,'models','forest');
forestFn = fullfile(forestDir, modelFnm);
if(exist([forestFn '.mat'], 'file'))
    load([forestFn '.mat']); 
else
    error('model not found');
end
%% Set model parameters
model.opts.multiscale=0;            % set to 0 to reproduce results in paper
model.opts.nTreesEval=8;            % set to 8 
model.opts.nThreads=8;              % max number threads for evaluation

%% Load testing data split (created during training -- see srfAffTrain_cornell.m)
testFP=load(fullfile(pwd, 'models', [modelFnm '_testDat.mat'])); testFP=testFP.testFP;
imgIds=strrep(testFP,'r.png','');
DepthFP=strcat(dataDir, imgIds, '.txt')';
ImFP=strcat(dataDir, testFP)';
ERes_cell=cell(length(ImFP),1);

%% Extract grasp detection response
parfor i=1:length(ImFP)
    I=imread(ImFP{i,1});
    [points, imPoints] = readGraspingPcd(DepthFP{i,1});
    D = zeros(size(I,2),size(I,1));
    D(imPoints) = points(:,3); D=D';
    [Nx, Ny, Nz] = surfnorm(D);
    DN=cat(3,Nx,Ny,Nz);    
    D=single(D)./1e2;
    I=cat(3,D,single(DN));
 
    E=affDetect_norm(I,model); 
    ERes_cell{i,1}=E;
end

% Save results for evaluation
save(fullfile(pwd, 'models', [modelFnm '_testRes.mat']), 'ERes_cell', '-v7.3');

%% Evaluate results
load(fullfile(pwd, 'models', [modelFnm '_testRes.mat']));
% get groundtruths
posFP=strcat(dataDir, imgIds, 'cpos.txt')';
negFP=strcat(dataDir, imgIds, 'cneg.txt')';
RecAccS = evaluateCornellRecognition_fn(ERes_cell,posFP,negFP);
DetAccS = evaluateCornellDetection_fn(ERes_cell,posFP);
fprintf('Cornell evaluation scores: Recognition | Detection -- %.3f | %.3f\n', RecAccS, DetAccS);

