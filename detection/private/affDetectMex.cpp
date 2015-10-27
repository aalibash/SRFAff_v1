/*********************************************************************************
* Detect affordance from forest -- helper mex function
* To compile, run compile.m
*
* Portions of this code are derived from:
* Structured Edge Detection Toolbox      Version 1.00
* Copyright 2013 Piotr Dollar.  [pdollar-at-microsoft.com] 
*
* Copyright (c) 2015 Ching L. Teo, University of Maryland College Park [cteo-at-cs.umd.edu]
* Licensed under the Simplified BSD License [see license.txt]
* Please email me if you find bugs, or have suggestions or questions!
**********************************************************************************/
#include <mex.h>
#include <math.h>
#include <matrix.h>
#include <stdlib.h>
#ifdef USEOMP
#include <omp.h>
#endif

typedef unsigned int uint32;
typedef unsigned short uint16;
#define min(x,y) ((x) < (y) ? (x) : (y))

// construct lookup array for mapping fids to channel indices
uint32* buildLookup( int *dims, int w ) {
  int c, r, z, n=w*w*dims[2]; uint32 *cids=new uint32[n]; n=0;
  for(z=0; z<dims[2]; z++) for(c=0; c<w; c++) for(r=0; r<w; r++)
    cids[n++] = z*dims[0]*dims[1] + c*dims[0] + r;
  return cids;
}

// [E,ind] = mexFunction(model,chns) - helper for affDetect_norm.m
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  // get inputs
  mxArray *model = (mxArray*) pr[0];
  float *chns = (float*) mxGetData(pr[1]);

  // extract relevant fields from model and options
  float *thrs = (float*) mxGetData(mxGetField(model,0,"thrs"));
  uint32 *fids = (uint32*) mxGetData(mxGetField(model,0,"fids"));
  uint32 *child = (uint32*) mxGetData(mxGetField(model,0,"child"));
  uint16 *eBins = (uint16*) mxGetData(mxGetField(model,0,"eBins"));
  uint32 *eBnds = (uint32*) mxGetData(mxGetField(model,0,"eBnds"));
  mxArray *opts = mxGetField(model,0,"opts");
  const int shrink = (int) mxGetScalar(mxGetField(opts,0,"shrink"));
  const int imWidth = (int) mxGetScalar(mxGetField(opts,0,"imWidth"));
  const int gtWidth = (int) mxGetScalar(mxGetField(opts,0,"gtWidth"));
  const int nChns = (int) mxGetScalar(mxGetField(opts,0,"nChns"));
  const uint32 nChnFtrs = (uint32) mxGetScalar(mxGetField(opts,0,"nChnFtrs"));
  const int stride = (int) mxGetScalar(mxGetField(opts,0,"stride"));
  const int nTreesEval = (int) mxGetScalar(mxGetField(opts,0,"nTreesEval"));
  int nThreads = (int) mxGetScalar(mxGetField(opts,0,"nThreads"));

  // get dimensions and constants
  const mwSize *chnsSize = mxGetDimensions(pr[1]);
  const int h = (int) chnsSize[0]*shrink;
  const int w = (int) chnsSize[1]*shrink;
  const mwSize *fidsSize = mxGetDimensions(mxGetField(model,0,"fids"));
  const int nTreeNodes = (int) fidsSize[0];
  const int nTrees = (int) fidsSize[1];
  const int h1 = (int) ceil(double(h-imWidth)/stride);
  const int w1 = (int) ceil(double(w-imWidth)/stride);
  const int h2 = h1*stride+gtWidth;
  const int w2 = w1*stride+gtWidth;
  const int chnDims[3] = {h/shrink,w/shrink,nChns};
  mwSize indDims[3] = {h1,w1,nTreesEval};
  mwSize outDims[3] = {h2,w2,1};

  // construct lookup tables
  uint32 *eids, *cids;
  eids = buildLookup( (int*)outDims, gtWidth );
  cids = buildLookup( (int*)chnDims, imWidth/shrink );

  // create outputs
  pl[0] = mxCreateNumericArray(3,outDims,mxSINGLE_CLASS,mxREAL);
  float *E = (float*) mxGetData(pl[0]);
  pl[1] = mxCreateNumericArray(3,indDims,mxUINT32_CLASS,mxREAL);
  uint32 *ind = (uint32*) mxGetData(pl[1]);

  // apply forest to all patches and store leaf inds
  #ifdef USEOMP
  nThreads = min(nThreads,omp_get_max_threads());
  #pragma omp parallel for num_threads(nThreads)
  #endif
  for( int c=0; c<w1; c++ ) for( int t=0; t<nTreesEval; t++ ) {
    for( int r0=0; r0<2; r0++ ) for( int r=r0; r<h1; r+=2 ) {
      int o = (r*stride/shrink) + (c*stride/shrink)*h/shrink;
      // select tree to evaluate
      int t1 = ((r+c)%2*nTreesEval+t)%nTrees; uint32 k = t1*nTreeNodes;
      while( child[k] ) {
        // compute feature
        uint32 f = fids[k]; float ftr;
        if( f<nChnFtrs ) ftr = chns[cids[f]+o]; 
	else continue;
        // compare ftr to threshold and move left or right accordingly
        if( ftr < thrs[k] ) k = child[k]-1; else k = child[k];
        k += t1*nTreeNodes;
      }
      // store leaf index and update affordance maps
      ind[ r + c*h1 + t*h1*w1 ] = k;
    }
  }

  // compute affordance maps
  for( int c0=0; c0<gtWidth/stride; c0++ ) {
    #ifdef USEOMP
    #pragma omp parallel for num_threads(nThreads)
    #endif
    for( int c=c0; c<w1; c+=gtWidth/stride ) {
      for( int r=0; r<h1; r++ ) for( int t=0; t<nTreesEval; t++ ) {
        uint32 k = ind[ r + c*h1 + t*h1*w1 ];
        float *E1 = E + (r*stride) + (c*stride)*h2;
        int b0=eBnds[k], b1=eBnds[k+1]; if(b0==b1) continue;
        for( int b=b0; b<b1; b++ ) E1[eids[eBins[b]]]++;
      }
    }
  }

  delete [] eids; delete [] cids;
}
