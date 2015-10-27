function BB_F=getBBF(gt_label)
% Estimates largest bounding box given the labeled ground truth location
%
% USAGE
%  BB_F = getBBF(gt_label)
%
% INPUT
%  gt_label         - [HxW] labeled image of ground truth
%
% OUTPUT
%  BB_F             - [1x4] bounding box [top-left x, top-left y, width, height]
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

gt_mask=gt_label>0;
RR=regionprops(gt_mask,'BoundingBox','area'); nObj=length(RR);
BB=[RR.BoundingBox]'; BB=reshape(BB,4,nObj); BB=BB';
BB(:,3)=BB(:,1)+BB(:,3); BB(:,4)=BB(:,2)+BB(:,4);
TL=min(BB(:,1:2),[],1); BR=max(BB(:,3:4),[],1);
BB_F = [TL, BR(1)-TL(1), BR(2)-TL(2)];
end
