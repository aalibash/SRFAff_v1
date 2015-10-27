function [chnsReg] = affChns_normal_cd( I, opts )
% Compute features for SRF afforances
% Uses precomputed depth, curvature and normals
%
% USAGE
%   [chnsReg] = affChns_normal_cd( I, opts )
%
% INPUT
%   I               - [HxWxD] rgb or grayscale image
%   opts            - parameters [see srfAffTrain.m]
% 
% OUTPUT
%   chnsReg         - [HxWxopts.nChns] feature "channels"
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!
%
% NOTE: opts.rgbd options. ONLY opts.rgbd=2 is supported. 
% 0: use 2D (RGB) features only, nTypes=1
% 1: use Depth features (depth+gradient+gradient mag) only, nTypes=1
% 2: use 3D features only (Depth + normals + curvatures), nTypes=3
% 3: use 2D (RGB) features + 3D features, nTypes=4

shrink=opts.shrink; nTypes=1; 
chns=cell(1,opts.nChns); k=0;
if(size(I,3)>3),
    if opts.rgbd==2
       nTypes=3; 
       Is={I(:,:,1),I(:,:,2:4),I(:,:,5:end)};           % {Depth,Normal,Curvature}
    elseif opts.rgbd>=3
        nTypes=4; 
        Is={I(:,:,1),I(:,:,2:4),I(:,:,5:7),I(:,:,8:end)}; %{Depth,RGB,Normal,Curvature}
    end
end
for t=1:nTypes
  if(nTypes>1), I=Is{t}; end
  if (t==1 && opts.rgbd==0)||(t==2 && opts.rgbd>=3) % Only convert RGB data
    if(size(I,3)==1), cs='gray'; else cs='luv'; end; 
    I=rgbConvert(I,cs);
  end  
  Ishrink=imResample(I,1/shrink); k=k+1; 
  if ~(t==2 && opts.rgbd==4) && ~(t==1)
    chns{k}=Ishrink;
  end
  % process over multiple scales
  for i = 1:2, s=2^(i-1);
    if(s==shrink), I1=Ishrink; 
    else I1=imResample(I,1/s);
    end
    I1 = convTri( I1, opts.grdSmooth );
    if t==1 % {DEPTH}
        [M,O] = gradientMag( I1, 0, opts.normRad, .01 );
        H = gradientHist( M, O, max(1,shrink/s), opts.nOrients, 0 );
        k=k+1; chns{k}=imResample(M,s/shrink);
        k=k+1; chns{k}=imResample(H,max(1,s/shrink));
    elseif t==3 && opts.rgbd==2 || t==4 && opts.rgbd>=3 % {Curvature+SI}
        CV1=nanmax(I1,[],3); CV2=nanmin(I1,[],3); % principle curvatures, CV1>CV2
        CV=sqrt((CV1.^2+CV2.^2)./2); % Curvedness
        % use region on CV which are "small" to smooth out their values
        % before computing SI
        CVMask=CV<prctile(CV(:),opts.CVpctile);
        CV1M=CV1; CV1M(~CVMask)=nan; CV1S=ndnanfilter(CV1M,'gausswin',[8/s 8/s], 1,[],{'replicate'},1);
        CV2M=CV2; CV2M(~CVMask)=nan; CV2S=ndnanfilter(CV2M,'gausswin',[8/s 8/s], 1,[],{'replicate'},1);
        CV1M=CV1; CV1M(CVMask)=nan; CV1SS=ndnanfilter(CV1M,'gausswin',[2/s 2/s], 1,[],{'replicate'},1);
        CV2M=CV2; CV2M(CVMask)=nan; CV2SS=ndnanfilter(CV2M,'gausswin',[2/s 2/s], 1,[],{'replicate'},1);

        SI=(-2/pi).*atan((CV2S+CV1S)./(CV1S-CV2S));
        SI2=(-2/pi).*atan((CV2SS+CV1SS)./(CV1SS-CV2SS));
        SI(isnan(SI))=SI2(isnan(SI)); % SHAPE INDEX
        k=k+1; chns{k}=imResample(SI,s/shrink);
        k=k+1; chns{k}=imResample(CV,s/shrink);
        CVMean=(CV1+CV2)/2; CVGauss=CV1.*CV2;
        k=k+1; chns{k}=imResample(CVMean,s/shrink);
        k=k+1; chns{k}=imResample(CVGauss,s/shrink);
    elseif t==2 && opts.rgbd>=3 % {RGB}
        [M,O] = gradientMag( I1, 0, opts.normRad, .01 );
        H = gradientHist( M, O, max(1,shrink/s), opts.nOrients, 0 );
        k=k+1; chns{k}=imResample(M,s/shrink);
        k=k+1; chns{k}=imResample(H,max(1,s/shrink));
    end
  end
end
chns=cat(3,chns{1:k}); assert(size(chns,3)==opts.nChns);
chnSm=opts.chnSmooth/shrink; if(chnSm>1), chnSm=round(chnSm); end
chnsReg=convTri(chns,chnSm);

end
