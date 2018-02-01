#include <stdlib.h>
#include <stdio.h>
#include "device_model.h"

int main(int argc, char** argv) {
	float id = device_model(0.2, 0.2, 0.2, 1);
	printf("%f\n", id);
	id = device_model(0.0, 0.0, 0.2, 1);
	printf("%f\n", id);
	id = device_model(0.1, 0.0, 0.2, 1);
	printf("%f\n", id);
	id = device_model(0.0, 0.1, 0.2, 1);
	printf("%f\n", id);
}