function DetAcc = evaluateCornellDetection_fn(ERes_cell,posFP)
% Returns detection accuracy for Cornell Grasping dataset: is the top scoring point near to
% one of the groundtruths? See README_cornell.txt for usage. 
%
% USAGE
%  DetAcc = evaluateCornellDetection_fn(ERes_cell,posFP)
%
% INPUT
%  ERes_cell    - {Rx1} cell of grasp detection results
%  posFP        - {Rx1} string cell of positive rectangles filenames (*cpos.txt) 
%
% OUTPUT
%  DetAcc       - Detection accuracy score
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions

ScoreT=[];
assert(length(ERes_cell)==length(posFP));

for i=1:length(posFP)
    % get data
    E=ERes_cell{i,1}; 
    posRects=load(posFP{i});
    CentroidsXY=[];
    posRMaskCC=false(size(E,1), size(E,2));
    
    if ~isempty(posRects)
         posRects=[posRects(~isnan(posRects(:,1)),1), posRects(~isnan(posRects(:,2)),2)];
         for aa=1:size(posRects,1)/4
            a1 = posRects((aa-1)*4+1:(aa-1)*4+4,:);
            posRMaskC=poly2mask(a1(:,1), a1(:,2),size(E,1),size(E,2));  
            posRMaskCC=or(posRMaskCC, posRMaskC);
            BB=regionprops(posRMaskC,'centroid');
            CentroidsXY=[CentroidsXY; BB.Centroid];
        end
    end
    Enorm=E; Enorm(~posRMaskCC)=0; Enorm=(Enorm - min(Enorm(:))) / ( max(Enorm(:)) - min(Enorm(:)) );
    [YMax,XMax]=find(Enorm==max(Enorm(:))); % top score
    DistP=pdist2([XMax,YMax], CentroidsXY);
    DistPI=find(DistP<=21); % we set 21 pixels as the distance threshold [see README_cornell.txt for a justification]
    if ~isempty(DistPI)
        ScoreT=[ScoreT; 1 1]; % hit
    else
        ScoreT=[ScoreT; 0 1]; % miss
    end

end
diffT = abs(ScoreT(:,1)-ScoreT(:,2));
DetAcc = 1 - sum(diffT)/size(ScoreT,1);

end