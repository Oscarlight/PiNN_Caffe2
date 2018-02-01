#ifndef _DEVICE_MODEL_H_   /* Include guard */
#define _DEVICE_MODEL_H_

// INPUTS: voltage in V, w in m, 
// OUTPUT: current in A
float device_model(
	const float vtg, 
	const float vbg, 
  	const float vds, 
  	const float w
);

#endif // _DEVICE_MODEL_H_