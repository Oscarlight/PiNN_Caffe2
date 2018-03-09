#ifndef _DEVICE_MODEL_H_   /* Include guard */
#define _DEVICE_MODEL_H_

// INPUTS: voltage in V, w in m, 
// OUTPUT: current in A
double device_model(
	const double vtg, 
	const double vbg, 
  	const double vds, 
  	const double w
);

#endif // _DEVICE_MODEL_H_
