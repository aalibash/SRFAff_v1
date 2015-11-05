% Evaluation script to load a trained SRF affordance model and 
% run it over the entire dataset -- highly parallelized
% Produces F^w_{\beta} scores reported in Table II of the paper.
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


function [] = script_evalAffordanceResRanked_cd(tID)

    clearvars -except tID; close all;

    %% set paths
    para.dir.code = pwd;  
    para.dir.data = 'data/Affordance_Part_Data/'; % location of dataset
    % precomputed data
    para.dir.cleanDepth='/media/data/projectData/affordance_structured_random_forest/umd_code/hmp/icra_2015_results/img_feats/depth/'; % (~13G)
    para.dir.cleanNorm='/media/data/projectData/affordance_structured_random_forest/umd_code/hmp/icra_2015_results/img_feats/normals/'; % (~38G)
    para.dir.cleanCurvature='/media/data/projectData/affordance_structured_random_forest/umd_code/hmp/icra_2015_results/img_feats/curvature/'; % (~9.3G)

    addpath(genpath(pwd)); addpath(para.dir.data);
    label_classes; 
    tS=2; tE=8; % ignore {background} labels
    meanTestV=[];

    %% process over all affordance labels
    % for tID=tS:tE
        % labels are zero-indexed so we minus 1
        target=affordance_label{tID};
        targetID=tID-1;
        modelFnm=['modelFinal_' target '_AF3_3Dp_N_S1']; %_AF_3Dp_F, _AF3_3Dp_N_S1
        test_str='test';
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
        model.opts.vggFeatDir='../hmp/icra_2015_results/vgg19_%s/';
        model.opts.bCleanDepth=0;
        model.opts.cropD={(40:469),(20:589)};

        dirSaveRes=fullfile(para.dir.data, 'results',[modelFnm '_RES']);
        TestResFN=fullfile(dirSaveRes, test_str, 'WFb_scores.mat');
        TestResFN_neg=fullfile(dirSaveRes, test_str, 'WFb_scores_neg.mat');
        %% Process dataset (return same test/train split)
        if ~exist(TestResFN,'file') || ~exist(TestResFN_neg,'file')
            if ~exist(dirSaveRes, 'dir'), mkdir(dirSaveRes); end;
            % Get test image filepaths
            toolsIds = dlmread('tool_type.txt');
            testGtDir = [model.opts.dataDir 'test/'];
            [testData] = getDataFP_cd(model, testGtDir, toolsIds);

            fprintf('Extracting testing results: %d positives | %d negatives\n', testData.nImgsP, testData.nImgsN);
            tic; extractAffordanceResFn_cd(model, testData, [dirSaveRes '/' test_str '/']); toc;
        end

        %% summarize results
        wFbTest=dlmread(TestResFN); meanTestV=[meanTestV;nanmean(wFbTest)];
        wFbTest_n=dlmread(TestResFN_neg); meanTestV=[meanTestV;nanmean(wFbTest_n)];
        wFbTestA=[wFbTest;wFbTest_n];
        fprintf('Averaged test WFb for %s: %f\n',modelFnm, nanmean(wFbTestA));
    % end
    fprintf('Mean Test: %f\n', mean(meanTestV));
    
end
