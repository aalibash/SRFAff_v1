% Script to demonstrate affordance detection via a trained SRF model.
% See the README and script_trainSRFAff.m for details on generating the
% affordance model. Precomputed models for all seven affordances are found in:
% "models/forest/precomputed_models/". Copy them into "models/forest/" 
% to use them here. 
% NOTE: Run "detection/private/compile.m" first before running this script
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
addpath(genpath(pwd));
%% load SRF affordance model and data 
bCleanDepth = 0; % 1: use precomputed depth/normals/curvature, 0: compute fast normals/curvature
target='scoop';%{'contain','cut','grasp','pound','scoop','support','wrap-grasp'};
label_classes; 
targetID=find(strcmp(affordance_label,target)==1)-1;
if bCleanDepth
    model_type='_AF3_3Dp_N_S1'; 
else
    model_type='_AF_3Dp_F';
end

modelFnm=['modelFinal_' target model_type];
forestDir = fullfile(pwd,'models','forest');
forestFn = fullfile(forestDir, modelFnm);
if(exist([forestFn '.mat'], 'file'))
  load([forestFn '.mat']); 
else
    error('model not found');
end
assert(targetID==model.opts.targetID);

%% SRF model parameters
model.opts.bCleanDepth=bCleanDepth; % 1: use precomputed depth/normals/curvature, 0: compute fast normals/curvature
model.opts.multiscale=0;            % detect over several scales (slower)
model.opts.nTreesEval=8;            % for top speed set nTreesEval=1
model.opts.nThreads=8;              % max number cpu threads for inference
%% read in data and process
if ~model.opts.bCleanDepth
%     rgbFN='data/00001_rgb.jpg'; depthFN='data/00001_depth.png';
    rgbFN='data/Affordance_Part_Data/test/bowl/bowl_01/bowl_01_00000001_rgb.jpg'; 
    depthFN='data/Affordance_Part_Data/test/bowl/bowl_01/bowl_01_00000001_depth.png'; 
    RGB=imread(rgbFN); D=single(imread(depthFN))./1e3;
    % [optional] resize to make things go a  bit faster...
    D=imresize(D,0.5); RGB=imresize(RGB,0.5); 
    % compute normals
    pcloud=depthtocloud(double(D)); DN = single(pcnormal(pcloud));
    D=single(D); RGB=im2single(RGB); 
    I=cat(3,D,DN); %{Depth,Normal}
    E=affDetect_norm(I,model);
else % load precomputed data
    dataFN='bowl_01_00000001'; %'mug_13_00000010'; %'ladle_05_00000019'; %'turner_05_00000067'; 
    dataFNF=dataFN(1:end-9); dataFNFD=strtok(dataFNF,'_');
    rgbFN=['data/Affordance_Part_Data/test/' dataFNFD '/' dataFNF '/' dataFN '_rgb.jpg']; 
    RGB=imread(rgbFN);
    depthFN=['data/depth_clean/' dataFN '.mat']; D=load(depthFN); D=D.depth_clean;
    curvatureFN=['data/curvature/' dataFN '.mat']; CV=load(curvatureFN); CV=CV.curvature;
    normalFN=['data/normals/' dataFN '.mat']; DN=load(normalFN); DN=DN.normals;
    % resize RGB
    RGB=im2uint8(imresize(RGB(model.opts.cropD{1}, model.opts.cropD{2},:),0.5));
    D=single(D); RGB=im2single(RGB); CV=single(CV);
    I=cat(3,D,DN,CV); % {Depth, Normal, Curvature}
    E=affDetect_norm(I,model);    
end
 
%% display affordance detection results
figure(1), imshow(RGB); title('Input RGB');
figure(2), imshow(E); title(sprintf('%s affordance detection',target));
