function [E] = affDetect_norm( I, model )
% Detects affordance using trained SRF model
% 
% USAGE
%  [E] = affDetect_norm( I, model )
%
% INPUT
%  I            - [HxWxD] rgb or grayscale image
%  model        - SRF affordance model [see srfAffTrain.m]
%
% OUTPUT
%  E            - [HxW] affordance confidence map (normalized)
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

% get parameters
opts=model.opts; opts.nTreesEval=min(opts.nTreesEval,opts.nTrees);
opts.stride=max(opts.stride,opts.shrink); model.opts=opts;

if( opts.multiscale )
  % if multiscale run affDetect_norm multiple times
  ss=2.^(-1:1); k=length(ss); siz=size(I);
  model.opts.multiscale=0; model.opts.nms=0; Es=0;
  for i=1:k, s=ss(i); 
    I1=imResample(I,s);
    [Es1]=affDetect_norm(I1,model);
    Es=Es+imResample(Es1,siz(1:2));
  end; Es=Es/k;
  
else
  % pad image, making divisible by 4
  sizOrig=size(I); r=opts.imWidth/2; p=[r r r r];
  p([2 4])=p([2 4])+mod(4-mod(sizOrig(1:2)+2*r,4),4);
  I = imPad(I,p,'symmetric');
  
  % compute features and apply forest to image
  if ~opts.bCleanDepth
        chnsReg = affChns_normal( I, opts );
  else
        chnsReg = affChns_normal_cd( I, opts );
  end
  Es = affDetectMex(model,chnsReg);

  % normalize and finalize affordance map
  t=opts.stride^2/opts.gtWidth^2/opts.nTreesEval;
  r=opts.gtWidth/2;
  Es=Es(1+r:sizOrig(1)+r,1+r:sizOrig(2)+r,:)*t; Es=convTri(Es,1);
end

% return final affordance map
E=Es;
end


