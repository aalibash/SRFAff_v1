function model = srfAffTrain( varargin )
% Trains a structured random forest (SRF) affordance model
%
% USAGE
%  opts     = srfAffTrain()
%  model    = srfAffTrain( opts )
%
% INPUTS
%  opts       - parameters (struct or name/value pairs)
%   (1) model parameters:
%   .imWidth        - [32] width of image patches
%   .gtWidth        - [16] width of ground truth patches
%   (2) tree parameters:
%   .nPos           - [1e5] number of positive patches per tree
%   .nNeg           - [1e5] number of negative patches per tree
%   .nImgs          - [inf] maximum number of images to use for training
%   .nTrees         - [8] number of trees in forest to train
%   .fracFtrs       - [1/2] fraction of features to use to train each tree
%   .minCount       - [1] minimum number of data points to allow split
%   .minChild       - [8] minimum number of data points allowed at child nodes
%   .maxDepth       - [64] maximum depth of tree
%   .discretize     - ['pca'] options include 'pca' and 'kmeans'
%   .nSamples       - [256] number of samples for clustering structured labels
%   .nClasses       - [2] number of classes (clusters) for binary splits
%   .split          - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   .targetID       - [-1] affordance target ID (see label_classes.m)
%   .treeTrainID    - [-1] vector containing the specific set of trees to train (see README.txt)
%   (3) feature parameters:
%   .nOrients       - [4] number of orientations per gradient scale
%   .grdSmooth      - [0] radius for image gradient smoothing (using convTri)
%   .chnSmooth      - [2] radius for reg channel smoothing (using convTri)
%   .normRad        - [4] gradient normalization radius (see gradientMag)
%   .shrink         - [2] amount to shrink channels
%   .rgbd           - [2] Ablation flags. 0: RGB only, 1: Depth only, 2: Depth+CV+SI
%   .bCleanDepth    - [0] 0: compute fast normals+curvature from raw depth, 1: use precomputed depth+normals+curvature  
%   .cropD          - [] croping parameters for precomputed data
%   .CVpctile       - [50] select regions smaller than CVpctile percentile of curvature values for estimating shape-index (see affChn_normals_cd.m) 
%   (4) detection parameters (can be altered after training):
%   .stride         - [2] stride at which to compute affordance region (resolution)
%   .multiscale     - [1] if true run affordance detection over multiple scales
%   .nTreesEval     - [8] number of trees to evaluate per patch
%   .nThreads       - [8] number of threads for evaluation of trees
%   (5) other parameters:
%   .seed           - [1] seed for random stream (for reproducibility)
%   .useParfor      - [0] if true train trees in parallel (memory intensive: ~12GB/Tree)
%   .modelDir       - ['models/'] target directory for storing models
%   .modelFnm       - ['modelFinal'] SRF model filename
%   .dataDir        - ['data/Affordance_Part_Data/'] location of training dataset 
%   .cleanDepthDir  - ['data/depth_clean/'] location of precomputed depth
%   .cleanNormDir   - ['data/normals/'] location of precomputed normals
%   .cleanCurvatureDir - [data/curvature/] location of precomputed curvature
%   .posSkip        - [1] number of positive training images to skip, 1: do not skip
%   .negSkip        - [1] number of negative training images to skip, 1: do not skip
%
% OUTPUTS
%  model      - trained SRF affordance detector with the following fields
%   .opts           - input parameters and constants
%   .thrs           - [nNodes x nTrees] threshold corresponding to each fid
%   .fids           - [nNodes x nTrees] feature ids for each node
%   .child          - [nNodes x nTrees] index of child for each node
%   .count          - [nNodes x nTrees] number of data points at each node
%   .depth          - [nNodes x nTrees] depth of each node
%   .eBins          - data structure for storing affordance regions
%   .eBnds          - data structure for storing affordance regions
%
% Portions of this code are derived from:
% Structured Edge Detection Toolbox Version 1.0
% Copyright (c) 2013 Piotr Dollar. [pdollar-at-microsoft.com] 
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

%% get default parameters
dfs={'imWidth',32, 'gtWidth',16, 'nPos',50000, 'nNeg',50000, ...
  'nImgs',inf, 'nTrees',8, 'fracFtrs',1/2, 'minCount',1, 'minChild',8, ...
  'maxDepth',64, 'discretize','pca', 'nSamples', 256,'nClasses', 2, ...
  'split','gini', 'targetID', -1, 'treeTrainID', -1, 'nOrients',4, 'grdSmooth',0,...
  'chnSmooth',2, 'normRad',4, 'shrink',2, 'rgbd',2, 'bCleanDepth', 0, 'cropD', [],...
  'CVpctile', 50, 'stride',2, 'multiscale',1,...
  'nTreesEval',8, 'nThreads',8, 'seed',1, 'useParfor',0, 'modelDir','models/',...
  'modelFnm','modelFinal', 'dataDir','data/Affordance_Part_Data/',...
  'cleanDepthDir', 'data/depth_clean/', 'cleanNormDir', 'data/normals/',...
  'cleanCurvatureDir', 'data/curvature/', 'vggFeatDir','../hmp/icra_2015_results/vgg19_%s/',...
  'hogFeatDir','../hmp/icra_2015_results/hog/','posSkip', 1, 'negSkip', 1};
opts = getPrmDflt(varargin,dfs,1);
if(nargin==0), model=opts; return; end % return default parameters if no arguments

%% if SRF affordance model exists load it and return
forestDir = [opts.modelDir '/forest/'];
forestFn = [forestDir opts.modelFnm];
if(exist([forestFn '.mat'], 'file')), load([forestFn '.mat']); return; end

%% compute constant parameters and store in opts
nTrees=opts.nTrees; shrink=opts.shrink; opts.nPos=round(opts.nPos); opts.nNeg=round(opts.nNeg);
opts.nTreesEval=min(opts.nTreesEval,nTrees); opts.stride=max(opts.stride,shrink);
imWidth=opts.imWidth; gtWidth=opts.gtWidth; %imWidth=round(max(gtWidth,imWidth)/shrink/2)*shrink*2;
opts.imWidth=imWidth; opts.gtWidth=gtWidth;

%% set up feature dimensions
nChnsGrad=(opts.nOrients+1)*2; nChnsColor=3; nChnsNorm=0;
if(opts.rgbd==1), nChnsColor=1; end % only depth (2D)
if(opts.rgbd==2), nChnsColor=0; nChnsNorm=3; nChnsGrad=(opts.nOrients+1+4)*2; end % depth + normal + curvature + SI + CV
nChns = nChnsGrad+nChnsColor+nChnsNorm; 
if opts.bCleanDepth, nChns = nChns + 2; end;
if(opts.rgbd==4), nChns=128; end;
if(opts.rgbd==5), nChns=32; end;
opts.nChns = nChns;
opts.nChnFtrs = imWidth*imWidth*nChns/shrink/shrink;
opts.nTotFtrs=opts.nChnFtrs;
disp(opts);

%% prepare training of decision trees
if opts.treeTrainID > 0
    % count number of target images
    trnGtDir = [opts.dataDir 'train/'];
    tools_names = dir(trnGtDir); isD=[tools_names(:).isdir];
    tools_names={tools_names(isD).name};
    tools_names(ismember(tools_names,{'.','..'}))=[]; tools_names=tools_names';
    toolsIds = dlmread('tool_type.txt'); % list of tools with their main affordance types
    assert(length(tools_names)==length(toolsIds));
    [toolsPos,~]=find(toolsIds==opts.targetID); % tools that have this AF
    fprintf('positive tools: %s\n', tools_names{toolsPos});

    % get positive and negative data filepaths
    [nImgsP,imgP_fp,gtP_fp,rgbP_fp,normP_fp,curveP_fp,vggP_fp]=...
        getDataFilePaths(tools_names, toolsPos, trnGtDir, opts);

    [toolsNeg,~]=find(toolsIds~=opts.targetID);
    [nImgsN,imgN_fp,gtN_fp,rgbN_fp,normN_fp,curveN_fp,vggN_fp]=...
        getDataFilePaths(tools_names, toolsNeg, trnGtDir, opts);

    fprintf('number of positive/negative target images: %d | %d\n',...
        length(1:opts.posSkip:nImgsP), length(1:opts.negSkip:nImgsN));
    opts.nImgsP=nImgsP; opts.nImgsN=nImgsN; opts.posFN=imgP_fp; opts.negFN=imgN_fp; 
    opts.posRGB=rgbP_fp; opts.negRGB=rgbN_fp; opts.labelPosFN=gtP_fp; opts.labelNegFN=gtN_fp;
    opts.normP=normP_fp; opts.normN=normN_fp; opts.curveP=curveP_fp; opts.curveN=curveN_fp;
    opts.vggP=vggP_fp; opts.vggN=vggN_fp; opts.hogP=strrep(vggP_fp,'vgg19_%s','hog');
    opts.hogN=strrep(vggN_fp,'vgg19_%s','hog');

    % estimate feature size (saves memory later for training)
        disp('estimating ftrs size...');
    if opts.bCleanDepth
        tic;
        kc=estimateFtrsSize_cd(nImgsP, opts.labelPosFN, opts.posFN, opts.normP,...
            opts.curveP, opts.nPos, opts.nNeg, opts);
        toc;
    else
        tic;
        kc=estimateFtrsSize(nImgsP, opts.labelPosFN, opts.posFN, opts.nPos,...
            opts.nNeg, opts);
        toc;
    end

    fprintf('ftrs size: %d\n',kc);
    opts.kc=kc;
end
% generate stream for reproducibility of model
stream=RandStream('mrg32k3a','Seed',opts.seed);

%% train nTrees random trees (can be trained with parfor if *enough* memory)
if (opts.treeTrainID > 0)
    trainV=opts.treeTrainID;
    if(opts.useParfor)
        if ~opts.bCleanDepth
            parfor i=trainV(1):trainV(end), trainTree(opts,stream,i); end
        else
            parfor i=trainV(1):trainV(end), trainTree_cd(opts,stream,i); end
        end
    else
        if ~opts.bCleanDepth
            for i=trainV(1):trainV(end), trainTree(opts,stream,i); end
        else
            for i=trainV(1):trainV(end), trainTree_cd(opts,stream,i); end
        end
    end        
end

%% train SRF affordance forest after all trees are trained
if opts.treeTrainID == -1
    % accumulate trees and merge into final model
    treeFn = [opts.modelDir '/tree/' opts.modelFnm '_tree'];
    for i=1:nTrees
      t=load([treeFn int2str2(i,3) '.mat'],'tree'); t=t.tree;
      if(i==1), trees=t(ones(1,nTrees)); else trees(i)=t; end
    end
    nNodes=0; for i=1:nTrees, nNodes=max(nNodes,size(trees(i).fids,1)); end
    model.opts=opts; Z=zeros(nNodes,nTrees,'uint32');
    model.thrs=zeros(nNodes,nTrees,'single');
    model.fids=Z; model.child=Z; model.count=Z; model.depth=Z;
    model.eBins=zeros(nNodes*nTrees*gtWidth*gtWidth,1,'uint16');
    model.eBnds=Z; k=0;
    for i=1:nTrees, tree=trees(i); nNodes1=size(tree.fids,1);
      model.fids(1:nNodes1,i) = tree.fids;
      model.thrs(1:nNodes1,i) = tree.thrs;
      model.child(1:nNodes1,i) = tree.child;
      model.count(1:nNodes1,i) = tree.count;
      model.depth(1:nNodes1,i) = tree.depth;
      % store compact representation of affordance regions
      for j=1:nNodes
       % saves large segments near boundaries in leaf nodes, results in clean precise edges [better]
        if(j>nNodes1 || tree.child(j)), E=0; else
        E=logical(tree.hs(:,:,j)-1); end
        eBins=uint32(find(E)-1); k1=k+length(eBins);
        model.eBins(k+1:k1)=eBins; k=k1; model.eBnds(j,i)=k;
      end
    end
    model.eBnds=[0; model.eBnds(:)]; model.eBins=model.eBins(1:k);
    % save model
    if(~exist(forestDir,'dir')), mkdir(forestDir); end
    save([forestFn '.mat'], 'model', '-v7.3');
else
    model=[]; % return empty SRF affordance model, only decision trees are trained
end

end % end srfAffTrain() 

function trainTree( opts, stream, treeInd )
% Train a single decision tree for SRF affordance model
% Computes curvature and normals on the fly (fast)

% location of ground truth
trnImgDirP = opts.posFN; rgbImgDirP = opts.posRGB; 
trnGtDirP = opts.labelPosFN; nImgsP=opts.nImgsP; 

% extract commonly used options
imWidth=opts.imWidth; imRadius=imWidth/2; 
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
nChns=opts.nChns; nTotFtrs=opts.nTotFtrs;
nPos=opts.nPos; nNeg=opts.nNeg; shrink=opts.shrink;

% finalize setup
treeDir = [opts.modelDir '/tree/'];
treeFn = [treeDir opts.modelFnm '_tree'];
if(exist([treeFn int2str2(treeInd,3) '.mat'],'file')), return; end
fprintf('\n-------------------------------------------\n');
fprintf('Training tree %d of %d\n',treeInd,opts.nTrees); 
tStart=clock;

% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream();
set(stream,'Substream',treeInd);
RandStream.setGlobalStream( stream );

% collect positive and negative patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs))); 
k=opts.kc; ftrs = zeros(k,length(fids),'single');
labels = zeros(gtWidth,gtWidth,k,'uint8'); k = 0;
tid = ticStatus('Collecting data',1,1);

%% collect positives first
nPA=length(1:opts.posSkip:nImgsP); ic=0;
for i=1:opts.posSkip:nImgsP
    load(trnGtDirP{i}); % load gt_label
    D=imread(trnImgDirP{i}); DMask=D>0;
    % read in RGB too
    RGB=imread(rgbImgDirP{i});
    [DDX,DDY,DDZ]=surfnorm(single(D)); DN=cat(3,DDX,DDY,DDZ);
    BB_F=getBBF(gt_label); gtL=gt_label;
    % read VGG features
    if(opts.rgbd==4), VGG=vggload(opts.vggP{i},{'2_2'}); end;
    if(opts.rgbd==5), HOG=hogload(opts.hogP{i}); end;
    % add in features accordingly
    D=single(D)./1e3; RGB=im2single(RGB);
    
    % Ablations. NOTE: ONLY opts.rgbd=2 is supported.
    if opts.rgbd == 0, I=RGB; end               %{RGB}
    if opts.rgbd == 1, I=D; end                 %{Depth}
    if opts.rgbd == 2, I=cat(3,D,DN); end       %{Depth,Normal}
    if opts.rgbd == 3, I=cat(3,D,RGB,DN); end   %{Depth,RGB,Normal}
    if opts.rgbd == 4, I=cat(3,VGG); end;       %{VGG}
    if opts.rgbd == 5, I=cat(3,HOG); end;       %{HOG}
    
    siz=size(I);
    p=zeros(1,4); p([2 4])=mod(4-mod(siz(1:2),4),4);
    if(any(p)), I=imPad(I,p,'symmetric'); end
    % compute features 
    chnsReg = affChns_normal(I,opts); 
    
    %% sample positives and negatives
    xy=[]; k1=0; B=false(siz(1),siz(2));
    B(shrink:shrink:end,shrink:shrink:end)=1;
    B([1:max(gtRadius,imRadius) end-max(gtRadius,imRadius):end],:)=0;
    B(:,[1:max(gtRadius,imRadius) end-max(gtRadius,imRadius):end])=0;
    
    M=gtL==opts.targetID;
    M=bwperim(M); M(bwdist(M)<gtRadius)=1; % Detect positives along boundaries [better]

    [y,x]=find(M.*B.*DMask); 
    k2=min(length(y),ceil(nPos/nPA));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 

    % take negatives within BB_F + outside [better]
    bbM=false(size(M)); 
    bbM(ceil(BB_F(2)):floor(BB_F(2)+BB_F(4)),...
        ceil(BB_F(1)):floor(BB_F(1)+BB_F(3)))=true;
    % half from real negative parts
    M=(gtL>0 & gtL~=opts.targetID);
    [y,x]=find(M.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 
    % half from background
    [y,x]=find(~bbM.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 

    if(k1>size(ftrs,1)-k), k1=size(ftrs,1)-k; xy=xy(1:k1,:); end
    
    %% crop patches and ground truth labels
    psReg=zeros(imWidth/shrink,imWidth/shrink,nChns,k1,'single');
    lbls=zeros(gtWidth,gtWidth,k1,'uint8'); 
    gtLL=gtL==opts.targetID; gtLL=uint8(gtLL)+1;
    % psSim=psReg; 
    ri=imRadius/shrink; rg=gtRadius;
    for j=1:k1, xy1=xy(j,:); xy2=xy1/shrink;
        psReg(:,:,:,j)=chnsReg(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);
        t=gtLL(xy1(2)-rg+1:xy1(2)+rg,xy1(1)-rg+1:xy1(1)+rg);
        [~,~,t]=unique(t); % Suppress annotations with no boundary pixels [better]    
        lbls(:,:,j)=reshape(t,gtWidth,gtWidth);
    end
    % compute features and store
    ftrs1=[reshape(psReg,[],k1)'];
    if ~isempty(ftrs1)
        ftrs(k+1:k+k1,:)=ftrs1(:,fids); labels(:,:,k+1:k+k1)=lbls;
    end
    k=k+k1; 
    if(k==size(ftrs,1)), tocStatus(tid,1); break; end
    ic=ic+1; tocStatus(tid,ic/nPA);
end
if(k<size(ftrs,1)), ftrs=ftrs(1:k,:); labels=labels(:,:,1:k); end

%% Train affordance SRF classifier
pTree=struct('minCount',opts.minCount, 'minChild',opts.minChild, ...
  'maxDepth',opts.maxDepth, 'H',opts.nClasses, 'split',opts.split);
labels=mat2cell2(labels,[1 1 k]);
pTree.discretize=@(hs,H) discretize(hs,H,opts.nSamples,opts.discretize);
tree=forestTrain(ftrs,labels,pTree); tree.hs=cell2array(tree.hs);
tree.fids(tree.child>0) = fids(tree.fids(tree.child>0)+1)-1;
if(~exist(treeDir,'dir')), mkdir(treeDir); end
save([treeFn int2str2(treeInd,3) '.mat'],'tree'); e=etime(clock,tStart);
fprintf('Training of tree %d complete (time=%.1fs).\n',treeInd,e);
RandStream.setGlobalStream( streamOrig );

% NOTE: you can choose to collect pure negative patches, but this was not
% implemented for the current results reported.
% 
% % collect negatives next
% trnImgDirN = opts.negFN; rgbImgDirN = opts.negRGB;
% trnGtDirN=opts.labelNegFN; nImgsN=opts.nImgsN; 
% nNA=length(1:opts.negSkip:nImgsN); 
% for i=1:nNA
%  % TODO: add code to sample from negative patches
% end

end % end trainTree()

function kc=estimateFtrsSize(nImgsP, trnGtDirP, trnImgDirP, nPos, nNeg, opts)
% Preallocate memory for features (efficiency)

shrink=opts.shrink;
imWidth=opts.imWidth; imRadius=imWidth/2;
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
% collect positives first
nPA=length(1:opts.posSkip:nImgsP);
kC=zeros(nPA,1);
vI_PA=1:opts.posSkip:nImgsP;
parfor pp=1:nPA
    i=vI_PA(pp); 
    gt_label=load(trnGtDirP{i}); gt_label=gt_label.gt_label; % load gt_label
    % crop depth and labels as well  
    D=imread(trnImgDirP{i}); DMask=D>0;
    BB_F=getBBF(gt_label);
    gtL=gt_label;
    
    % add in features accordingly
    D=single(D)./1e3; siz=size(D);
    % sample positives and negatives
    k1=0; B=false(siz(1),siz(2));
    B(shrink:shrink:end,shrink:shrink:end)=1;
    B([1:imRadius end-imRadius:end],:)=0;
    B(:,[1:imRadius end-imRadius:end])=0;
    
    M=gtL==opts.targetID;
    M=bwperim(M); M(bwdist(M)<gtRadius)=1; % Detect positives along boundaries [better]

    [y,~]=find(M.*B.*DMask); 
    k2=min(length(y),ceil(nPos/nPA));
    k1=k1+k2; 

    % take negatives within BB_F + outside. [better]
    bbM=false(size(M)); 
    bbM(ceil(BB_F(2)):floor(BB_F(2)+BB_F(4)),...
        ceil(BB_F(1)):floor(BB_F(1)+BB_F(3)))=true;
    % half from real negative parts
    M=(gtL>0 & gtL~=opts.targetID);
    [y,~]=find(M.*B);
    k2=min(length(y),ceil(nNeg/nPA/2)); 
    k1=k1+k2; 
    % half from background
    [y,~]=find(~bbM.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    k1=k1+k2; 

    kC(pp,1)=k1;    
end
% reconstruct all data
kc=sum(kC);
end % end estimateFtrsSize()

function trainTree_cd( opts, stream, treeInd )
% Train a single decision tree for SRF affordance model
% For precomputed normals/depth/curvature

% setup data filepaths
trnImgDirP = opts.posFN; 
rgbImgDirP = opts.posRGB; trnGtDirP = opts.labelPosFN; 
normDirP = opts.normP; curveDirP=opts.curveP; 
nImgsP=opts.nImgsP;

% extract commonly used options
imWidth=opts.imWidth; imRadius=imWidth/2;
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
nChns=opts.nChns; nTotFtrs=opts.nTotFtrs;
nPos=opts.nPos; nNeg=opts.nNeg; shrink=opts.shrink;
kc=opts.kc;

% finalize setup
treeDir = [opts.modelDir '/tree/'];
treeFn = [treeDir opts.modelFnm '_tree'];
if(exist([treeFn int2str2(treeInd,3) '.mat'],'file')), return; end
fprintf('\n-------------------------------------------\n');
fprintf('Training tree %d of %d\n',treeInd,opts.nTrees); 
tStart=clock;

% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream(); set(stream,'Substream',treeInd);
RandStream.setGlobalStream( stream );

% collect positive and negative patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs)));
k=kc;
ftrs = zeros(k,length(fids),'single');
labels = zeros(gtWidth,gtWidth,k,'uint8'); k = 0;
tid = ticStatus('Collecting data',1,1);

%% collect positives first
nPA=length(1:opts.posSkip:nImgsP); ic=0;
for i=1:opts.posSkip:nImgsP
    load(trnGtDirP{i}); % load gt_label
    normalFP=normDirP{i}; % normal file
    curveFP=curveDirP{i}; % curvature file
    
    if ~exist(normalFP,'file') 
        continue;
    else % load precomputed normals
        DD=load(normalFP); DN=DD.normals; 
    end
    
    if ~exist(curveFP,'file') 
        continue;
    else % load precomputed normals
        CC=load(curveFP); CV=CC.curvature; 
    end
    D=load(trnImgDirP{i}); D=D.depth_clean; DMask=D>0;
    RGB=imread(rgbImgDirP{i});
    gtL=double(imresize(uint8(gt_label(opts.cropD{1}, opts.cropD{2})),0.5));  % crop labels
    RGB=im2uint8(imresize(RGB(opts.cropD{1}, opts.cropD{2},:),0.5));
    BB_F=getBBF(gtL);
    assert(sum(size(gtL) - size(D))==0); % same size
    assert(sum(size(RGB) - size(DN))==0);
    % add in features accordingly
    D=single(D); RGB=im2single(RGB); CV=single(CV);
    
    % Ablation options. NOTE: only opts.rgbd == 2 is currently supported.
    if opts.rgbd == 0, I=RGB; end                   %{RGB}
    if opts.rgbd == 1, I=D; end                     %{Depth}
    if opts.rgbd == 2, I=cat(3,D,DN,CV); end        %{Depth,Normal,Curvature}: reported in paper
    if opts.rgbd == 3, I=cat(3,D,RGB,DN,CV); end    %{Depth,RGB,Normal,Curvature}
    
    siz=size(I);
    p=zeros(1,4); p([2 4])=mod(4-mod(siz(1:2),4),4);
    if(any(p)), I=imPad(I,p,'symmetric'); end   
    % compute features 
    chnsReg = affChns_normal_cd(I,opts);
    
    %% sample positives and negatives
    xy=[]; k1=0; B=false(siz(1),siz(2));
    B(shrink:shrink:end,shrink:shrink:end)=1;
    B([1:imRadius end-imRadius:end],:)=0;
    B(:,[1:imRadius end-imRadius:end])=0;
    
    M=gtL==opts.targetID;
    M=bwperim(M); M(bwdist(M)<gtRadius)=1; % Detect positives along boundaries [better]
    
    [y,x]=find(M.*B.*DMask); 
    k2=min(length(y),ceil(nPos/nPA));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 
    
    % take negatives within BB_F + outside [better]
    bbM=false(size(M)); 
    bbM(ceil(BB_F(2)):floor(BB_F(2)+BB_F(4)),...
        ceil(BB_F(1)):floor(BB_F(1)+BB_F(3)))=true;
    % half from real negative parts
    M=(gtL>0 & gtL~=opts.targetID);
    [y,x]=find(M.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 
    % half from background
    [y,x]=find(~bbM.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)]; k1=k1+k2; 
   
    if(k1>size(ftrs,1)-k), k1=size(ftrs,1)-k; xy=xy(1:k1,:); end
    
    %% crop patches and ground truth labels
    psReg=zeros(imWidth/shrink,imWidth/shrink,nChns,k1,'single');
    lbls=zeros(gtWidth,gtWidth,k1,'uint8'); 
    gtLL=gtL==opts.targetID; gtLL=uint8(gtLL)+1;
    ri=imRadius/shrink; rg=gtRadius;
    for j=1:k1, xy1=xy(j,:); xy2=xy1/shrink;
        psReg(:,:,:,j)=chnsReg(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);
        t=gtLL(xy1(2)-rg+1:xy1(2)+rg,xy1(1)-rg+1:xy1(1)+rg);
        [~,~,t]=unique(t); % Suppress annotations with no boundary pixels [better]    
        lbls(:,:,j)=reshape(t,gtWidth,gtWidth);
    end
    % compute features and store
    ftrs1=reshape(psReg,[],k1)';
    if ~isempty(ftrs1)
        ftrs(k+1:k+k1,:)=ftrs1(:,fids); labels(:,:,k+1:k+k1)=lbls;
    end
    k=k+k1; 
    if(k==size(ftrs,1)), tocStatus(tid,1); break; end
    ic=ic+1; tocStatus(tid,ic/nPA);
end
if(k<size(ftrs,1)), ftrs=ftrs(1:k,:); labels=labels(:,:,1:k); end

%% Train SRF affordance classifier
pTree=struct('minCount',opts.minCount, 'minChild',opts.minChild, ...
  'maxDepth',opts.maxDepth, 'H',opts.nClasses, 'split',opts.split);
labels=mat2cell2(labels,[1 1 k]);
pTree.discretize=@(hs,H) discretize(hs,H,opts.nSamples,opts.discretize);
tree=forestTrain(ftrs,labels,pTree); tree.hs=cell2array(tree.hs);
tree.fids(tree.child>0) = fids(tree.fids(tree.child>0)+1)-1;
if(~exist(treeDir,'dir')), mkdir(treeDir); end
save([treeFn int2str2(treeInd,3) '.mat'],'tree'); e=etime(clock,tStart);
fprintf('Training of tree %d complete (time=%.1fs).\n',treeInd,e);
RandStream.setGlobalStream( streamOrig );

% NOTE: you can choose to collect pure negative patches, but this was not
% implemented for the current results reported.
% 
% % collect negatives next
% trnImgDirN = opts.negFN; rgbImgDirN = opts.negRGB;
% trnGtDirN=opts.labelNegFN; nImgsN=opts.nImgsN; 
% nNA=length(1:opts.negSkip:nImgsN); 
% for i=1:nNA
%  % TODO: add code to sample from negative patches
% end

end % end trainTree_cd()

function kc=estimateFtrsSize_cd(nImgsP,trnGtDirP, trnImgDirP, normDirP, curveDirP, nPos, nNeg, opts)
% Preallocate memory for features (efficiency)
% For precomputed data: depth/normals/curvature

shrink=opts.shrink;
imWidth=opts.imWidth; imRadius=imWidth/2; 
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
% collect positives first
nPA=length(1:opts.posSkip:nImgsP);
kC=zeros(nPA,1); vI_PA=1:opts.posSkip:nImgsP;
parfor pp=1:nPA
    i=vI_PA(pp);
    gt_label=load(trnGtDirP{i}); gt_label=gt_label.gt_label; % load gt_label
    normalFP=normDirP{i}; % normal file
    curveFP=curveDirP{i}; % curvature file
    if ~exist(normalFP,'file') 
        %warning('precomputed normal file missing'); 
        continue; % skip
    end
    
    if ~exist(curveFP,'file') 
        %warning('precomputed curvature file missing'); 
        continue; % skip
    end
    
    D=load(trnImgDirP{i}); D=D.depth_clean; DMask=D>0;
    gtL=double(imresize(uint8(gt_label(opts.cropD{1}, opts.cropD{2})),0.5));  % crop labels
    BB_F=getBBF(gtL);
    assert(sum(size(gtL) - size(D))==0); % same size
    % add in features accordingly
    D=single(D); siz=size(D);
    % sample positives and negatives
    k1=0; B=false(siz(1),siz(2));
    B(shrink:shrink:end,shrink:shrink:end)=1;
    B([1:imRadius end-imRadius:end],:)=0;
    B(:,[1:imRadius end-imRadius:end])=0;
    
    M=gtL==opts.targetID;
    M=bwperim(M); M(bwdist(M)<gtRadius)=1; % Detect positives along boundaries [better]

    [y,~]=find(M.*B.*DMask); 
    k2=min(length(y),ceil(nPos/nPA));
    k1=k1+k2; 
   
    % take negatives within BB_F + outside... (better)
    bbM=false(size(M)); 
    bbM(ceil(BB_F(2)):floor(BB_F(2)+BB_F(4)),...
        ceil(BB_F(1)):floor(BB_F(1)+BB_F(3)))=true;
    % half from real negative parts
    M=(gtL>0 & gtL~=opts.targetID);
    [y,~]=find(M.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    k1=k1+k2; 
    % half from background
    [y,~]=find(~bbM.*B);
    k2=min(length(y),ceil(nNeg/nPA/2));
    k1=k1+k2; 
 
    kC(pp,1)=k1;
    
end
% reconstruct all data
kc=sum(kC);
end % end estimateFtrsSize_cd()

function [nImgs,img_fp,gt_fp,rgb_fp,norm_fp,curve_fp,vgg_fp]=...
    getDataFilePaths(tools_names, toolsSet, trnGtDir, opts)
% Determine filepaths of data

nImgs=0; img_fp={}; gt_fp={}; rgb_fp={}; norm_fp={}; curve_fp={}; vgg_fp={};
for pp=1:length(toolsSet)
    gtD=[trnGtDir tools_names{toolsSet(pp)} '/'];
    gtN=dir(gtD); isD=[gtN(:).isdir];
    gtN={gtN(isD).name};
    gtN(ismember(gtN,{'.','..'}))=[]; gtN=gtN';
    for gg=1:length(gtN)
        if ~opts.bCleanDepth
            imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
            imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
            img_fp=[img_fp ;imgIds];
        else
            imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
            imgIds=strrep(imgIds,'_depth.png','');
            imgIds=strcat(opts.cleanDepthDir, imgIds, '.mat')'; % use cleaned depth
            img_fp=[img_fp ;imgIds];
            imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
            imgIds=strrep(imgIds,'_depth.png','');
            imgIds=strcat(opts.cleanNormDir, imgIds, '.mat')'; % use cleaned norm
            norm_fp=[norm_fp ;imgIds];
            imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
            imgIds=strrep(imgIds,'_depth.png','');
            imgIds=strcat(opts.cleanCurvatureDir, imgIds, '.mat')'; % use cleaned norm
            curve_fp=[curve_fp ;imgIds];
        end
        imgIds=dir([gtD, gtN{gg}  '/*.png']); imgIds={imgIds.name};
        imgIds=strrep(imgIds,'_depth.png','');
        imgIds=strcat(opts.vggFeatDir, imgIds, '.mat')'; % use cleaned depth
        vgg_fp=[vgg_fp;imgIds];
        imgIds=dir([gtD, gtN{gg}  '/*.jpg']); imgIds={imgIds.name};
        imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
        rgb_fp=[rgb_fp ;imgIds];
        imgIds=dir([gtD, gtN{gg}  '/*_label.mat']); imgIds={imgIds.name};
        imgIds=strcat([gtD, gtN{gg} '/'], imgIds)';
        gt_fp=[gt_fp; imgIds];
        nImgs=nImgs+length(imgIds);
    end    
end
end % end getDataFilePaths()

function [hs,seg] = discretize( segs, nClasses, nSamples, type )
% Convert a set of segmentations into a set of labels in [1,nClasses]

persistent cache; w=size(segs{1},1); assert(size(segs{1},2)==w);
if(~isempty(cache) && cache{1}==w), [~,is1,is2]=deal(cache{:}); else
  % compute all possible lookup inds for w x w patches
  is=1:w^4; is1=floor((is-1)/w/w); is2=is-is1*w*w; is1=is1+1;
  kp=is2>is1; is1=is1(kp); is2=is2(kp); cache={w,is1,is2};
end
% compute n binary codes zs of length nSamples
nSamples=min(nSamples,length(is1)); kp=randperm(length(is1),nSamples);
n=length(segs); is1=is1(kp); is2=is2(kp); zs=false(n,nSamples);
for i=1:n, zs(i,:)=segs{i}(is1)==segs{i}(is2); end
zs=bsxfun(@minus,zs,sum(zs,1)/n); zs=zs(:,any(zs,1));
if(isempty(zs)), hs=ones(n,1,'uint32'); seg=segs{1}; return; end
% find most representative seg (closest to mean)
[~,ind]=min(sum(zs.*zs,2)); seg=segs{ind};
% apply PCA to reduce dimensionality of zs
U=pca(zs'); d=min(5,size(U,2)); zs=zs*U(:,1:d);
% discretize zs by clustering or discretizing pca dimensions
d=min(d,floor(log2(nClasses))); hs=zeros(n,1);
for i=1:d, hs=hs+(zs(:,i)<0)*2^(i-1); end
[~,~,hs]=unique(hs); hs=uint32(hs);
if(strcmpi(type,'kmeans'))
  nClasses1=max(hs); C=zs(1:nClasses1,:);
  for i=1:nClasses1, C(i,:)=mean(zs(hs==i,:),1); end
  hs=uint32(kmeans2(zs,nClasses,'C0',C,'nIter',1));
end
% optionally display different types of hs
for i=1:0, figure(i); montage2(cell2array(segs(hs==i))); end
end % end discretize()

