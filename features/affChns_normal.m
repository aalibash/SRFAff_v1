function chnsReg = affChns_normal( I, opts )
% Compute features for SRF afforances
% Computes curvature and normals on the fly (fast version)
%
% USAGE
%   [chnsReg] = affChns_normal( I, opts )
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
if opts.rgbd==4, chnsReg=imresize(I,1/opts.shrink,'nearest'); return; end;
if opts.rgbd==5, chnsReg=imresize(I,1/opts.shrink,'nearest'); return; end;
shrink=opts.shrink; nTypes=1; 
chns=cell(1,opts.nChns); k=0;
if(size(I,3)>3),
    if opts.rgbd==2
        nTypes=2; 
        Is={I(:,:,1),I(:,:,2:end)};            %{Depth,Normal}
    elseif opts.rgbd>=3
        nTypes=3; 
        Is={I(:,:,1),I(:,:,2:4),I(:,:,5:end)}; %{Depth,RGB,Normal}
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
    if(s==shrink), I1=Ishrink; else I1=imResample(I,1/s); end
    I1 = convTri( I1, opts.grdSmooth );
    if t==1 % depth + surface curvature
        [M,O] = gradientMag( I1, 0, opts.normRad, .01 );
        H = gradientHist( M, O, max(1,shrink/s), opts.nOrients, 0 );
        k=k+1; chns{k}=imResample(M,s/shrink);
        k=k+1; chns{k}=imResample(H,max(1,s/shrink));
        if opts.rgbd > 1 % {process DEPTH}: curvature + SI
            sizz=size(I1); 
            [xx,yy]=meshgrid(1:sizz(2), 1:sizz(1));
            IM=I1==0; 
            [G,MC,PMax,PMin] = surfature(xx,yy,double(I1)); 
            G(IM)=0; MC(IM)=0; PMax(IM)=0; PMin(IM)=0; 
            PMax=real(PMax); PMin=real(PMin);
            SI=(-2/pi).*atan((PMax+PMin)./(PMax-PMin));  SI(isnan(SI))=0;
            CV=sqrt((PMax.^2+PMin.^2)./2);
            k=k+1; chns{k}=imResample(G,s/shrink);
            k=k+1; chns{k}=imResample(MC,s/shrink);
            k=k+1; chns{k}=imResample(SI,s/shrink);
            k=k+1; chns{k}=imResample(CV,s/shrink);
        end
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

