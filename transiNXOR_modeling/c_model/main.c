#include <stdlib.h>
#include <stdio.h>
#include "device_model.h"

int main(int argc, char** argv) {
    double vtg = 0.2;
    double vtb = 0.2;
    double vds = 0.2;
	double id = device_model(vtg, vtb, vds, 1);
	printf("%f\n", id);
	id = device_model(0.0, 0.0, 0.2, 1);
	printf("%f\n", id);
	id = device_model(0.1, 0.0, 0.2, 1);
	printf("%f\n", id);
	id = device_model(0.0, 0.1, 0.2, 1);
	printf("%f\n", id);
}
