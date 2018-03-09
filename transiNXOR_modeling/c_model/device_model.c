#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  include <mkl_cblas.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "device_model.h"


/* -------------------- HELPER FUNCTIONS -------------------- */
// Don't support batch mode
void fc(const int m_in, const int n_in, const int k_in, 
	const float *W_in, const float *I_in, const float *B_in, float* O_in) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m_in, n_in, k_in, 1.0, W_in, k_in, I_in, n_in, 0.0, O_in, n_in);
  cblas_saxpy(m_in, 1.0, B_in, 1, O_in, 1);
}

void matmul(const int m_in, const int n_in, const int k_in, 
	const float *W_in, const float *I_in, float *O_in) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m_in, n_in, k_in, 1.0, W_in, k_in, I_in, n_in, 0.0, O_in, n_in);
}

void add(const int n_in, float *a_in, float *b_in) {
	cblas_saxpy(n_in, 1.0, a_in, 1, b_in, 1);
}

void sig_act(float *a_in, const int len) {
	int i;
	for (i=0; i < len; i++)
		a_in[i] = (tanh(a_in[i]/2) + 1)/2;
}

void tanh_act(float *a, const int len) {
	int i;
	for (i=0; i < len; i++) {
		a[i] = tanh(a[i]);
	}
}

void print_array(float *a, const int len) {
	int i;
	for (i=0; i < len; i++)
		printf("%f, ", a[i]);
	printf("\n");
}


/* -------------------- DEVICE MODEL -------------------- */
double device_model( const double vtg_d, const double vbg_d, 
  const double vds_d, const double w_d
) {
    
    #include "device_param"
    // convert to float, cadence does not like floats
    const float vtg = (float) vtg_d;
    const float vbg = (float) vbg_d;
    const float vds = (float) vds_d;
    const float w = (float) w_d;

    float vg[2] = {(vtg-0.1)/0.1, (vbg-0.1)/0.1};
    float vd[1] = {vds/0.2};
    float sig_temp0[16]  = {0};
    float tanh_temp0[16] = {0};
    float inter_temp[16] = {0};
    float sig_temp1[16]  = {0};
    float tanh_temp1[16] = {0};

    // Layer 0
    fc(16, 1, 2, sig_fc_layer_0_w, vg, sig_fc_layer_0_b, sig_temp0);
    matmul(16, 1, 1, tanh_fc_layer_0_w, vd, tanh_temp0);
    fc(16, 1, 16, inter_embed_layer_0_w, tanh_temp0, inter_embed_layer_0_b, inter_temp);
    add(16, inter_temp, sig_temp0);
    sig_act(sig_temp0, 16);
    tanh_act(tanh_temp0, 16);

    // Layer 1
    fc(16, 1, 16, sig_fc_layer_1_w, sig_temp0, sig_fc_layer_1_b, sig_temp1);
    matmul(16, 1, 16, tanh_fc_layer_1_w, tanh_temp0, tanh_temp1);
    fc(16, 1, 16, inter_embed_layer_1_w, tanh_temp1, inter_embed_layer_1_b, inter_temp);
    add(16, inter_temp, sig_temp1);;
    sig_act(sig_temp1, 16);
    tanh_act(tanh_temp1, 16);
        
    // Layer 2
    fc(1, 1, 16, sig_fc_layer_2_w, sig_temp1, sig_fc_layer_2_b, sig_temp0);
    matmul(1, 1, 16, tanh_fc_layer_2_w, tanh_temp1, tanh_temp0);
    fc(1, 1, 1, inter_embed_layer_2_w, tanh_temp0, inter_embed_layer_2_b, inter_temp);
    add(1, inter_temp, sig_temp0);
    tanh_act(tanh_temp0, 1);
    sig_act(sig_temp0, 1);
    // Output   
    double ids = (double) (sig_temp0[0] * tanh_temp0[0] * 53.65093994 * w);
    return ids;

}

