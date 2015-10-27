% Script that compiles the affDetectMex helper function. Run this prior to calling scrip_runDetectionDemo.m
% NOTE: tested on 64-bit UNIX only, use in other OS is not guaranteed (but should work)
%
% Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
% Licensed under the Simplified BSD License [see license.txt]
% Please email me if you find bugs, or have suggestions or questions!

setenv('OMP_NUM_THREADS', '8');
archstr=computer('arch');
if isunix
    if strcmp(archstr,'glnxa64')
        mex affDetectMex.cpp -largeArrayDims -DUSEOMP -v CXXFLAGS='\$CXXFLAGS -fopenmp' LDFLAGS='\$LDFLAGS -fopenmp';
    else
        mex affDetectMex.cpp -compatibleArrayDims -DUSEOMP -v CXXFLAGS='\$CXXFLAGS -fopenmp' LDFLAGS='\$LDFLAGS -fopenmp';
    end
elseif ispc
    if strcmp(archstr,'pcwin64')
        mex affDetectMex.cpp -largeArrayDims -v COMPFLAGS="/openmp $COMPFLAGS";
    else
        mex affDetectMex.cpp -compatibleArrayDims -v COMPFLAGS="/openmp $COMPFLAGS";
    end
else
    error('Operating system not supported');
end
   
