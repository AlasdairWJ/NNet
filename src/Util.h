#ifndef __UTIL_H
#define __UTIL_H

#define PI 3.14159265359
#define TAU 6.28318530718

double random();

double ReLU(double x);
double ReLU_dx(double x);

double sigmoid(double x);
double sigmoid_dx(double x);

double randNorm();

void LoadCSV(const char* file, double* values, unsigned count);
void SaveCSV(const char* file, double* values, unsigned count, unsigned new_row_after = 0);

#endif