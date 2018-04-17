#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define EPS 1e-8

double random() {
	return rand() * (1.0 / RAND_MAX);
}

double ReLU(double x) {
	return x >= 0.0 ? x : 0.0;
}

double ReLU_dx(double x) {
	return x >= 0.0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_dx(double x) {
	double lx = sigmoid(x);
	return lx * (1.0 - lx);
}

double randNorm() {
	static int got = 0;
	static double z1;

	if (got) {
		got = 0;
		return z1;
	}

	double u1, u2, z0;
	do {
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while ( u1 <= 1e-8 );
	got = 1;

	u1 = sqrt(-2.0 * log(u1));
	u2 *= TAU;
	z0 = u1 * cos(u2);
	z1 = u1 * sin(u2);

	return z0;
}

void LoadCSV(const char* filename, double* values, unsigned count) {
	FILE* fileobj = fopen(filename, "r");

	for (unsigned n = 0; n < count; n++) {
		fscanf(fileobj, "%lf,", &values[n]);
	}

	fclose(fileobj);
}

void SaveCSV(const char* filename, double* values, unsigned count, unsigned new_row_after) {
	FILE* fileobj = fopen(filename, "r");

	for (unsigned n = 0; n < count; n++) {
		fscanf(fileobj, "%lf,", &values[n]);
		if (new_row_after) {
			if ((n+1) % new_row_after == 0) {
				fputc('\n', fileobj);
			}
		}
	}

	fclose(fileobj);
}