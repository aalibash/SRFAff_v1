function extractAffordanceResFn_ranked_cd(ModelsC, outData, dirSaveRes)
% Run affordance SRF detection over dataset, and saves results to file for
% evaluation. For ranked ground truth labels.
%
% USAGE
%  extractAffordanceResFn_ranked_cd(ModelsC, outData, dirSaveRes)
%
% INPUT
%  ModelsC              - {Mx1} cell of trained SRF affordance models (see script_trainSRFAff.m)
%  outData              - filepaths of data to be tested (see getDataFP_ranked_cd.m)
%  dirSaveRes           - directory where all results are to be saved
% 
% OUTPUT
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

if ~exist(dirSaveRes, 'dir'), mkdir(dirSaveRes); end;

[nImgsP, nImgsN, imgP_fp, gtP_fp,...
    rgbP_fp, normP_fp, curveP_fp, rankP_fp, imgN_fp, gtN_fp, rgbN_fp, normN_fp, curveN_fp, rankN_fp]=deal(outData.nImgsP, outData.nImgsN,outData.imgP_fp,outData.gtP_fp,...
    outData.rgbP_fp, outData.normP_fp, outData.curveP_fp, outData.rankP_fp, outData.imgN_fp,...
    outData.gtN_fp,outData.rgbN_fp,outData.normN_fp,outData.curveN_fp,outData.rankN_fp);

cropD=ModelsC{8,1}.opts.cropD;
% convert ranks to normalized weights
wVec=[1 2 4 8 16 32 64 128 256]; wVec= 1./wVec; wVec=wVec/sum(wVec);
wVec=wVec';

% extract positives first
disp('processing positives...');
if ~exist(fullfile(dirSaveRes, 'WFbRanked_scores_N.mat'), 'file')

WFbS_Pv=cell(nImgsP,1); % ranked over 7 affordances row 1(top), row 7 (worst)
parfor i=1:nImgsP
    I=[];
    if ~exist(imgP_fp{i,1},'file'), continue; end;
    D=load(imgP_fp{i,1}); 
    D=D.depth_clean; 
    RGB=imread(rgbP_fp{i,1}); 
    normT=load(normP_fp{i,1}); DN=normT.normals;
    if ~exist(curveP_fp{i,1},'file'), continue; end;
    CV=load(curveP_fp{i,1}); 
    CV=CV.curvature;
    % get BB_F from original labels
    GT_o=load(gtP_fp{i,1}); GT_o=GT_o.gt_label; 
    GT_o=imresize(uint8(GT_o(cropD{1},cropD{2})),0.5, 'nearest');  % crop labels
    BB_F=getBBF(GT_o);
    GT_rank=load(rankP_fp{i,1}); GT_rank=GT_rank.gt_label;
    RGB=im2uint8(imresize(RGB(cropD{1}, cropD{2},:),0.5));
    % eval on cropped data
    D=imcrop(D,BB_F); RGB=imcrop(RGB,BB_F);
    DN = imcrop(DN, BB_F); 
    CV1=imcrop(CV(:,:,1),BB_F); CV2=imcrop(CV(:,:,2),BB_F); CV=cat(3,CV1,CV2);
    WFbi=nan(7,1);
    
    % go through all affordances
    for rr=2:size(GT_rank,3)+1 
        model=ModelsC{rr,1}; GT=GT_rank(:,:,rr-1);
        % resize GT
        GT=imresize(uint8(GT(model.opts.cropD{1}, model.opts.cropD{2})),0.5, 'nearest');  % crop labels
        GT=imcrop(GT,BB_F);  

        D=single(D); RGB=im2single(RGB); CV=single(CV);
        if model.opts.rgbd == 0, I=RGB; end % {RGB}
        if model.opts.rgbd == 1, I=D; end %{Depth}
        if model.opts.rgbd == 2, I=cat(3,D,DN,CV); end %{Depth,Normal,Curvature}
        if model.opts.rgbd == 3, I=cat(3,D,RGB,DN,CV); end %{Depth,RGB,Normal}
        E=affDetect_norm(I,model);
        % apply rank-weighted WFb
        wFBs=nan(9,1);
        for gg=1:9 % there are 9 ranks
            gT=GT==gg;
            if sum(gT(:)>0), wFBs(gg,1)=WFb(double(E),gT); end
        end
        wFBss=wVec.*wFBs;
        WFbi(rr-1,1)=nansum(wFBss);
    end
    WFbS_Pv{i,1}=WFbi; 
end
% save scores 
WFbS_PvF=cell2mat(WFbS_Pv');
% write a temp file in case something goes wrong later
dlmwrite(fullfile(dirSaveRes, 'WFbRanked_scores_N_posTMP.mat'), WFbS_PvF);

%% 
disp('processing negatives...');
WFbS_Nv=cell(nImgsN,1);
parfor i=1:nImgsN
    I=[];
    if ~exist(imgN_fp{i,1},'file'), continue; end;
    D=load(imgN_fp{i,1}); 
    D=D.depth_clean; 
    RGB=imread(rgbN_fp{i,1}); 
    normT=load(normN_fp{i,1}); DN=normT.normals;
    if ~exist(curveN_fp{i,1},'file'), continue; end;
    CV=load(curveN_fp{i,1}); 
    CV=CV.curvature;
    % get BB_F from original labels
    GT_o=load(gtN_fp{i,1}); GT_o=GT_o.gt_label; 
    GT_o=imresize(uint8(GT_o(cropD{1},cropD{2})),0.5, 'nearest');  % crop labels
    BB_F=getBBF(GT_o);
    GT_rank=load(rankN_fp{i,1}); GT_rank=GT_rank.gt_label;
    RGB=im2uint8(imresize(RGB(cropD{1}, cropD{2},:),0.5));
    % eval on cropped data
    D=imcrop(D,BB_F); RGB=imcrop(RGB,BB_F);
    DN = imcrop(DN, BB_F); 
    CV1=imcrop(CV(:,:,1),BB_F); CV2=imcrop(CV(:,:,2),BB_F); CV=cat(3,CV1,CV2);
    WFbi=nan(7,1);
    
    % go through all affordances
    for rr=2:size(GT_rank,3)+1 
        model=ModelsC{rr,1}; GT=GT_rank(:,:,rr-1);
        % resize GT
        GT=imresize(uint8(GT(model.opts.cropD{1}, model.opts.cropD{2})),0.5, 'nearest');  % crop labels
        GT=imcrop(GT,BB_F);  %GT=GT==model.opts.targetID;

        D=single(D); RGB=im2single(RGB); CV=single(CV);
        if model.opts.rgbd == 0, I=RGB; end % {RGB}
        if model.opts.rgbd == 1, I=D; end %{Depth}
        if model.opts.rgbd == 2, I=cat(3,D,DN,CV); end %{Depth,Normal,Curvature}
        if model.opts.rgbd == 3, I=cat(3,D,RGB,DN,CV); end %{Depth,RGB,Normal}
        E=affDetect_norm(I,model);
        % apply rank-weighted WFb
        wFBs=nan(9,1);
        for gg=1:9 % there are 9 ranks
            gT=GT==gg;
            if sum(gT(:)>0), wFBs(gg,1)=WFb(double(E),gT); end
        end
        wFBss=wVec.*wFBs;
        WFbi(rr-1,1)=nansum(wFBss);
    end
    WFbS_Nv{i,1}=WFbi; 
    
end
% save scores 
WFbS_NvF=cell2mat(WFbS_Nv');
% write a temp file in case something goes wrong later
dlmwrite(fullfile(dirSaveRes, 'WFbRanked_scores_N_negTMP.mat'), WFbS_NvF);


%% Combine results and save
WFbS_PvFC=[WFbS_PvF, WFbS_NvF];
dlmwrite(fullfile(dirSaveRes, 'WFbRanked_scores_N.mat'), WFbS_PvFC);

end

end
