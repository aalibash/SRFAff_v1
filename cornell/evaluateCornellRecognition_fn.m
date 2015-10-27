function RecAcc = evaluateCornellRecognition_fn(ERes_cell,posFP,negFP)
% Returns recognition accuracy for Cornell Grasping dataset: are positive patches correctly
% recognized? See README_cornell.txt for usage. 
%
% USAGE
%  RecAcc = evaluateCornellRecognition_fn(ERes_cell,ImFP,posFP,negFP)
%
% INPUT
%  ERes_cell    - {Rx1} cell of grasp detection results
%  posFP        - {Rx1} string cell of positive rectangles filenames (*cpos.txt) 
%  negFP        - {Rx1} string cell of negative rectangles filenames (*cneg.txt) 
%
% OUTPUT
%  RecAcc       - Recognition accuracy score
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!


ScoreT=[];
assert(length(posFP)==length(negFP));
assert(length(ERes_cell)==length(posFP));

for i=1:length(posFP)
    % get data
    E=ERes_cell{i,1}; 
    posRects=load(posFP{i});
    negRects=load(negFP{i});
    posRMaskCC=false(size(E,1),size(E,2)); 
    negRMaskCC=posRMaskCC;
    
    % normalize correctly first
     if ~isempty(posRects)
        posRects=[posRects(~isnan(posRects(:,1)),1), posRects(~isnan(posRects(:,2)),2)];
        for aa=1:size(posRects,1)/4
            a1 = posRects((aa-1)*4+1:(aa-1)*4+4,:);
            posRMaskC=poly2mask(a1(:,1), a1(:,2),size(E,1),size(E,2));
            posRMaskCC=or(posRMaskCC,posRMaskC);
        end
    end
    if ~isempty(negRects)
        negS=[];
        negRects=[negRects(~isnan(negRects(:,1)),1), negRects(~isnan(negRects(:,2)),2)];
        for aa=1:size(negRects,1)/4
            a1 = negRects((aa-1)*4+1:(aa-1)*4+4,:);
            negRMaskC=poly2mask(a1(:,1), a1(:,2),size(E,1),size(E,2));
            negRMaskCC=or(negRMaskCC,negRMaskC);
        end
    end
    maskCC=or(posRMaskCC,negRMaskCC); Enorm=E; Enorm(~maskCC)=0;
    Enorm=(Enorm - min(Enorm(:))) / ( max(Enorm(:)) - min(Enorm(:)) );
    
    if ~isempty(posRects)
        posS=[];
        posRects=[posRects(~isnan(posRects(:,1)),1), posRects(~isnan(posRects(:,2)),2)];
        for aa=1:size(posRects,1)/4
            a1 = posRects((aa-1)*4+1:(aa-1)*4+4,:);
            posRMaskC=poly2mask(a1(:,1), a1(:,2),size(E,1),size(E,2));
        
            Ea1=Enorm; Ea1(~posRMaskC)=0; 
            if max(Ea1(:))>0.5, posS = [posS; 1 1]; % a hit
            else posS = [posS; 0 1]; % a miss
            end;
        end
    end    

    if ~isempty(negRects)
        negS=[];
        negRects=[negRects(~isnan(negRects(:,1)),1), negRects(~isnan(negRects(:,2)),2)];
        for aa=1:size(negRects,1)/4
            a1 = negRects((aa-1)*4+1:(aa-1)*4+4,:);
            negRMaskC=poly2mask(a1(:,1), a1(:,2),size(E,1),size(E,2));
         
            Ea1=Enorm; Ea1(~negRMaskC)=0;
            if mean(Ea1(:))>0.5, negS = [negS; 1 0]; % a miss
            else negS = [negS; 0 0]; % a hit
            end;
        end
    end
    ScoreC=[posS;negS];
    ScoreT=[ScoreT;ScoreC];    
end

diffT=abs(ScoreT(:,1)-ScoreT(:,2));
RecAcc= 1-sum(diffT)/size(ScoreT,1);

end