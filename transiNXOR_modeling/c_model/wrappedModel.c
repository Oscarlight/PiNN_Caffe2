/**
 *wrappedModel.c
 *interface betweeen c model and matlab
 *compile with mex wrappedModel.c device_model.c -lopenblas -I... -L...
 *-I should be cblas include folder and -L should be cblas lib folder
 *call as wrappedModel(vtg, vbg, vds, w)
 */

#include "mex.h"
#include "device_model.h"
#define SCALE_FACTOR 1e-6

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (nrhs != 4 || nlhs > 1) {
        mexErrMsgIdAndTxt("MATLAB:device_model", "4 inputs and one output required");
        return;
    }

    double vtg = mxGetScalar(prhs[0]);
    double vbg = mxGetScalar(prhs[1]);
    double vds = mxGetScalar(prhs[2]);
    double w = mxGetScalar(prhs[3]);
    
    plhs[0] = mxCreateDoubleMatrix((mwSize) 1, (mwSize) 1, mxREAL); // create output array

    double *res = mxGetPr(plhs[0]);
    *res = SCALE_FACTOR*device_model(vtg, vbg, vds, w); // actual function call
}
