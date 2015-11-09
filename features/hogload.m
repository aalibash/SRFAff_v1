function out=hogload(path)
    out = [];
    feat=load(path);
    feat=feat.feat;
    feat=imresize(feat,8,'nearest');
    out = cat(3,out,feat);
end