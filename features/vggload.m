function out=vggload(path,tag) % assumes first level is 2_2 (=240*320, half resolution == resolution for training)
    out = [];
    for lev = 1:numel(tag)
        feat=load(sprintf(path,tag{lev}));
        feat=feat.scores;
        feat=imresize(feat,2^lev,'nearest');
        out = cat(3,out,feat);
    end
end