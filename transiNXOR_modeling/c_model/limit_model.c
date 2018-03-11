// make transitions smooth around boundaries of regions that were modeled
// The behavior should be that the result is predicted by the neuran model within the
// trained data region and transitions asymtotically outside the trained region
// http://www.mos-ak.org/rome/talks/T15_Mijalkovic_MOS-AK_Rome.pdf
//

#include <stdio.h>

#define VDS_MAX 0.3
#define VTG_MAX 0.3
#define VBG_MAX 0.3

#define VDS_MIN -0.1
#define VTG_MIN -0.1
#define VBG_MIN -0.1



double device_model( const double vtg_d, const double vbg_d, const double vds_d, const double w_d);

static inline double my_max(double a, double b){
    return a > b ? a: b;
}

static inline double my_min(double a, double b){
    return a < b ? a : b;
}

double vlimit_model(double vtg, double vbg, double vds, double w){

    FILE *f = fopen("simlog.txt", "a");
    fprintf(f, "vtg = %G\t vbg = %G\t vds = %G \t", vtg, vbg, vds);

    vds = my_min(my_max(vtg, VDS_MIN), VDS_MAX);
    vbg = my_min(my_max(vbg, VBG_MIN), VBG_MAX);
    vtg = my_min(my_max(vtg, VTG_MIN), VTG_MAX);
    double ids = device_model(vtg, vbg, vds, w);
    fprintf(f, "ids = %G \n");
    fclose(f);
    return ids;
}

