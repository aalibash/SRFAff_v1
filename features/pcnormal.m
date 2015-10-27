function normal = pcnormal(pcloud) 
% compute surface normals by principle component analysis
%
% USAGE
%   normal = pcnormal(pcloud)
%
% INPUT 
%   pcloud      - [HxWxD] point cloud [see depthtocloud.m]
%
% OUTPUT
%   normal      - [HxWxD] estimated normals
%
% written by Liefeng Bo on 2012
% optimized by Xiaofeng Ren on 2012
% further optimizations by Ching L. Teo by adding in parfor loops

threshold = 0.01; % 1 centimeter
win = 5; % window size
mindata = 3;

[imh, imw, ddim] = size(pcloud); % pcloud is a 3D matrix
normal = zeros(size(pcloud));
parfor i = 1:imh
%for i = 1:imh
    for j = 1:imw
        minh = max(i - win,1);
        maxh = min(i + win,size(pcloud,1));
        minw = max(j - win,1);
        maxw = min(j + win,size(pcloud,2));
        index = abs(pcloud(minh:maxh,minw:maxw,3) - pcloud(i,j,3)) < pcloud(i,j,3)*threshold;
        pcdis = pcloud(minh:maxh,minw:maxw,1) - pcloud(i,j,1);
        pcdis = pcloud(minh:maxh,minw:maxw,2) - pcloud(i,j,2);
        pcdis = pcloud(minh:maxh,minw:maxw,3) - pcloud(i,j,3);
        pcdis = sqrt(sum(pcdis.^2,3));
        pcij = sqrt(sum(pcloud(i,j,:).^2));
        index = (pcdis < pcij*threshold); 
        if sum(index(:)) > mindata & pcij > 0 % a minimum number of points required
            wpc = reshape(pcloud(minh:maxh,minw:maxw,:), (maxh-minh+1)*(maxw-minw+1),3);
            subwpc = wpc(index(:),:);
            subwpc = subwpc - ones(size(subwpc,1),1)*(sum(subwpc)/size(subwpc,1));
            [coeff,~] = eig(subwpc'*subwpc); 
            normal(i,j,:) = coeff(:,1)';
        end
    end
end

dd = sum(pcloud.*normal,3 );
normal = normal.*repmat(sign(dd),[1 1 3]);


