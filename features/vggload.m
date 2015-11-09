function out=vggload(path,tag) % assumes first level is 2_2 (=240*320, half resolution == resolution for training)
    out = [];
    for idx = 1:numel(tag)
		if(strcmp(tag{idx},'2_2'))	lev=1;
		elseif(strcmp(tag{idx},'3_4')) lev=2;
		elseif(strcmp(tag{idx},'4_4')) lev=3;
		elseif(strcmp(tag{idx},'5_4')) lev=4;
		end
        feat=load(sprintf(path,tag{idx}));
        feat=feat.scores;
        feat=imresize(feat,2^lev,'nearest');
        out = cat(3,out,feat);
    end
end
