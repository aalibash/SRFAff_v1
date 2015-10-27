function outData = getDataFP_ranked_cd(ModelsC, trnGtDir, toolsIds)
% Gets data filepaths in preparation for evaluation with ranked groundtruth
%
% USAGE
%  outData = getDataFP_ranked_cd(ModelsC, trnGtDir, toolsIds)
%
% INPUT
%  ModelsC 	- {Mx1} cell of trained SRF affordance models (see script_trainSRFAff.m)
%  trnGtDir 	- location of ground truth 
%  toolsIds 	- mapping of tool IDs with their affordances (see tool_type.txt)
%
% OUTPUT
%  outData 	- output structure containing data of filepaths for evaluation
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!


nImgsP=0; imgP_fp={}; gtP_fp={}; rgbP_fp={}; normP_fp={}; curveP_fp={};
rankP_fp={};
nImgsN=0; imgN_fp={}; gtN_fp={}; rgbN_fp={}; normN_fp={}; curveN_fp={};
rankN_fp={};

for k=1:length(ModelsC)
    model=ModelsC{k,1};
    if isempty(model), continue; end
    tools_names = dir(trnGtDir); isD=[tools_names(:).isdir];
    tools_names={tools_names(isD).name};
    tools_names(ismember(tools_names,{'.','..'}))=[]; tools_names=tools_names';
    assert(length(tools_names)==length(toolsIds));
    [toolsPos,~]=find(toolsIds==model.opts.targetID); % tools that have this AF
    fprintf('positive tools: %s\n', tools_names{toolsPos});

    for pp=1:length(toolsPos)
        gtD=[trnGtDir tools_names{toolsPos(pp)} '/'];
        gtN=dir(gtD); isD=[gtN(:).isdir];
        gtN={gtN(isD).name};
        gtN(ismember(gtN,{'.','..'}))=[]; gtN=gtN';
        parfor gg=1:length(gtN)
            if ~model.opts.bCleanDepth
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
                imgP_fp=[imgP_fp ;imgIds];
            else
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanDepthDir, imgIds, '.mat')'; % use cleaned depth
                imgP_fp=[imgP_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanNormDir, imgIds, '.mat')'; % use cleaned norm
                normP_fp=[normP_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanCurvatureDir, imgIds, '.mat')'; % use cleaned norm
                curveP_fp=[curveP_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.rankDir, imgIds, '_label_rank.mat')'; % use cleaned norm
                rankP_fp=[rankP_fp ;imgIds];

            end
            imgIds=dir([gtD, gtN{gg}  '/*.jpg']); imgIds={imgIds.name};
            imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
            rgbP_fp=[rgbP_fp ;imgIds];
            imgIds=dir([gtD, gtN{gg}  '/*_label.mat']); imgIds={imgIds.name};
            imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
            gtP_fp=[gtP_fp; imgIds];
            nImgsP=nImgsP+length(imgIds);
        end    
    end

    [toolsNeg,~]=find(toolsIds~=model.opts.targetID);
    for pp=1:length(toolsNeg)
        gtD=[trnGtDir tools_names{toolsNeg(pp)} '/'];
        gtN=dir(gtD); isD=[gtN(:).isdir];
        gtN={gtN(isD).name};
        gtN(ismember(gtN,{'.','..'}))=[]; gtN=gtN';
        parfor gg=1:length(gtN)
            if ~model.opts.bCleanDepth
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
                imgN_fp=[imgN_fp ;imgIds];
            else
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanDepthDir, imgIds, '.mat')'; % use cleaned depth
                imgN_fp=[imgN_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanNormDir, imgIds, '.mat')'; % use cleaned norm
                normN_fp=[normN_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.cleanCurvatureDir, imgIds, '.mat')'; % use cleaned norm
                curveN_fp=[curveN_fp ;imgIds];
                imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
                imgIds=strrep(imgIds,'_depth.png','');
                imgIds=strcat(model.opts.rankDir, imgIds, '_label_rank.mat')'; % use cleaned norm
                rankN_fp=[rankN_fp ;imgIds];
            end
            imgIds=dir([gtD, gtN{gg}  '/*.jpg']); imgIds={imgIds.name};
            imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
            rgbN_fp=[rgbN_fp ;imgIds];
            imgIds=dir([gtD, gtN{gg}  '/*_label.mat']); imgIds={imgIds.name};
            imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
            gtN_fp=[gtN_fp; imgIds];
            nImgsN=nImgsN+length(imgIds);
        end    
    end
    
end

% save output structure
outData=[];
[outData.nImgsP, outData.nImgsN,outData.imgP_fp,outData.gtP_fp,...
    outData.rgbP_fp, outData.normP_fp, outData.curveP_fp, outData.rankP_fp, outData.imgN_fp,...
    outData.gtN_fp,outData.rgbN_fp,outData.normN_fp, outData.curveN_fp, outData.rankN_fp]=deal(nImgsP, nImgsN, imgP_fp, gtP_fp,...
    rgbP_fp, normP_fp,curveP_fp, rankP_fp,imgN_fp, gtN_fp, rgbN_fp, normN_fp, curveN_fp, rankN_fp);

end
