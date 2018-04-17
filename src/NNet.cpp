#include "NNet.h"
#include "Util.h"

#include <cstdio>
#include <cmath>

double* NNet::Params() const {
	return this->params;
}

unsigned NNet::NumParams() const {
	return this->num_params;
}

unsigned NNet::DataSize() const {
	return this->data_size;
}

unsigned NNet::NumLabels() const {
	return this->num_labels;
}

NNet::NNet(unsigned* layer_sizes, unsigned num_layers) {
	this->data_size = layer_sizes[0];
	this->num_layers = num_layers;

	this->layer_sizes = new unsigned[num_layers];
	this->num_weights = 0;
	this->num_biases = 0;
	this->num_neurons = 0;

	unsigned last = layer_sizes[0];
	this->layer_sizes[0] = last;
	for (unsigned l = 1; l < num_layers; l++) {
		this->layer_sizes[l] = layer_sizes[l];

		this->num_biases += layer_sizes[l];
		this->num_weights += layer_sizes[l] * last;
		last = layer_sizes[l];
		this->num_neurons += last;
	}
	this->num_labels = last;

	this->num_params = this->num_weights + this->num_biases;
	this->params = new double[this->num_params];
	this->weights = this->params;
	this->biases = this->params + this->num_weights;
}

NNet::~NNet() {
	delete[] this->params;
	delete[] this->layer_sizes;
}

void NNet::Randomise() const {
	for (unsigned i = 0; i < this->num_params; i++) {
		this->params[i] = randNorm();
	}
}
	
void NNet::Save(const char* filaname) const {
	FILE* f = fopen(filaname, "wb");
	if (f == NULL) {
		return;
	}
	fwrite(this->params, sizeof(double), this->num_params, f);
	fclose(f);
}

void NNet::Load(const char* filaname) const {
	FILE* f = fopen(filaname, "rb");
	if (f == NULL) {
		return;
	}
	fread(this->params, sizeof(double), this->num_params, f);
	fclose(f);
}

void NNet::Classify(double* yhat, unsigned* classes, unsigned N) const {
	unsigned i, n, n0;
	for (n = 0; n < N; n++) {
		n0 = n * this->num_labels;
		classes[n] = 0;
		for (i = 1; i < this->num_labels; i++) {
			if (yhat[n0 + i] > yhat[n0 + classes[n]]) {
				classes[n] = i;
			}
		}
	}
}

double* NNet::YHat(double* neurons, unsigned N) const {
	return neurons + (2 * this->num_neurons - this->num_labels) * N;
}

double NNet::Cost(double* yhat, double* labels, unsigned N) const {
	double total = 0.0;
	unsigned n, i, ni;
	for (n = 0; n < N; n++) {
		for (i = 0; i < this->num_labels; i++) {
			ni = n * this->num_labels + i;
			total -= labels[ni] * log(yhat[ni]) + (1.0 - labels[ni]) * log(1.0 - yhat[ni]);
		}
	}
	return total / N;
}

double* NNet::Forward(double* data, unsigned N) const {
	double* neurons = new double[2 * this->num_neurons * N];

	double* prev_A = data;
	double* Z = neurons;
	double* A;

	double* W = this->weights;
	double* B = this->biases;

	unsigned last_layer_size = this->data_size;
	unsigned layer_size;

	unsigned l, n, i, j, ni, nj, ji;
	unsigned L = this->num_layers - 1;

	double maxval, total;

	for (l = 1; l <= L; l++) {
		layer_size = this->layer_sizes[l];
		A = Z + layer_size * N;

		for (n = 0; n < N; n++) {
			for (i = 0; i < layer_size; i++) {
				ni = n * layer_size + i;
				Z[ni] = B[i];
				for (j = 0; j < last_layer_size; j++) {
					ji = j * layer_size + i;
					nj = n * last_layer_size + j;
					Z[ni] += prev_A[nj] * W[ji];
				}

				if (l != L) {
					A[ni] = sigmoid(Z[ni]);
				}
			}

			if (l == L) {
				ni = n * layer_size;
				maxval = Z[ni];
				for (i = 1; i < layer_size; i++) {
					ni = n * layer_size + i;
					if (Z[ni] > maxval) {
						maxval = Z[ni];
					}
				}

				total = 0.0;
				for (i = 0; i < layer_size; i++) {
					ni = n * layer_size + i;
					A[ni] = exp(Z[ni] - maxval);
					total += A[ni];
				}

				for (i = 0; i < layer_size; i++) {
					ni = n * layer_size + i;
					A[ni] /= total;
				}
			}
		}

		prev_A = A;
		W += layer_size * last_layer_size;
		B += layer_size;
		Z += 2 * layer_size * N;

		last_layer_size = layer_size;
	}

	return neurons;
}

double* NNet::Back(double* data, double* neurons, double* labels, unsigned N) const {
	double* delta_params = new double[this->num_params];
	double* dB = delta_params + this->num_params;
	double* dW = delta_params + this->num_weights;

	double* delta_source = new double[this->num_neurons * N];

	unsigned n, i, j, ni, nj, ji;

	double* Z = neurons + 2 * this->num_neurons * N;
	double* A = Z - this->num_labels * N;

	double* W = this->weights + this->num_weights;

	unsigned next_layer_size;
	unsigned layer_size = this->num_labels;

	double* next_delta = delta_source + this->num_neurons * N;
	double* delta;

	double den;

	unsigned L = this->num_layers - 1;
	for (unsigned l = L; l > 0; l--) {
		delta = next_delta - layer_size * N;
		
		Z -= 2 * layer_size * N;

		if (l == L) { // dJ / dZ_nj^L
			for (n = 0; n < N; n++) {
				for (j = 0; j < layer_size; j++) {
					nj = n * layer_size + j;
					delta[nj] = 0.0;
					for (i = 0; i < layer_size; i++) {
						ni = n * layer_size + i;

						if (i == j) {
							delta[nj] += A[ni] - labels[ni];
						} else {
							den = 1.0 - A[ni];
							if (den < 1e-6) den = 1e-6;
							delta[nj] -= A[nj] * (A[ni] - labels[ni]) / den;
						}
					}
				}
			}
		} else { // dJ / dZ_nj^l
			for (n = 0; n < N; n++) {
				for (j = 0; j < layer_size; j++) {
					nj = n * layer_size + j;
					delta[nj] = 0.0;
					for (i = 0; i < next_layer_size; i++) {
						ni = n * next_layer_size + i;
						ji = j * next_layer_size + i;
						delta[nj] += next_delta[ni] * W[ji];
					}
					delta[nj] *= sigmoid_dx(Z[nj]);
				}
			}
		}

		next_layer_size = layer_size;
		layer_size = this->layer_sizes[l - 1];

		if (l != 1) {
			A = Z - next_layer_size * N;
		} else {
			A = data;
		}

		W -= layer_size * next_layer_size;
		dW -= layer_size * next_layer_size;
		dB -= next_layer_size;

		for (i = 0; i < next_layer_size; i++) {
			for (j = 0; j < layer_size; j++) {
				ji = j * layer_size + i;

				dW[ji] = 0.0;
				for (n = 0; n < N; n++) {
					ni = n * next_layer_size + i;
					nj = n * layer_size + j;
					dW[ji] += delta[ni] * A[nj];
				}

				dW[ji] /= N;
			}

			dB[i] = 0.0;
			for (n = 0; n < N; n++) {
				ni = n * next_layer_size + i;
				dB[i] += delta[ni];
			}
			dB[i] /= N;
		}

		next_delta = delta;
	}

	delete[] delta_source;

	return delta_params;
}


void NNet::PrintNeurons(double* data, double* neurons, unsigned N) const {
	unsigned l, i, n, ni;
	for (n = 0; n < N; n++) {
		for (i = 0; i < this->data_size; i++) {
			ni = n * this->data_size + i;
			printf(" % 8.5lf ", data[ni]);
		}
		putchar('\n');
	}
	putchar('\n');

	double* Z = neurons;
	double* A;

	unsigned layer_size;
	for (l = 1; l < this->num_layers; l++) {
		layer_size = this->layer_sizes[l];
		A = Z + layer_size * N;

		for (n = 0; n < N; n++) {
			for (i = 0; i < layer_size; i++) {
				ni = n * layer_size + i;
				printf(" % 8.5lf ", Z[ni]);
			}
			putchar('\n');
		}
		putchar('\n');

		for (n = 0; n < N; n++) {
			for (i = 0; i < layer_size; i++) {
				ni = n * layer_size + i;
				printf(" % 8.5lf ", A[ni]);
			}
			putchar('\n');
		}
		putchar('\n');

		Z += 2 * layer_size * N;
	}
}

void NNet::PrintParams(double* params) const {
	unsigned l, j, i, ji;

	double* W = this->weights;
	double* B = this->biases;
	unsigned layer_size;
	unsigned prev_layer_size = this->data_size;
	for (l = 1; l < this->num_layers; ++l) {
		layer_size = this->layer_sizes[l];
		for (j = 0; j < prev_layer_size; ++j) { 
			for (i = 0; i < layer_size; ++i) {
				ji = j * layer_size + i;
				printf(" % 12.8lf ", W[ji]);
			}
			putchar('\n');
		}
		for (i = 0; i < layer_size; ++i) {
			printf("  ------------");
		}
		putchar('\n');
		for (i = 0; i < layer_size; ++i) {
			printf(" % 12.8lf ", B[i]);
		}
		puts("\n");

		W += layer_size * prev_layer_size;
		B += layer_size;

		prev_layer_size = layer_size;
	}
}

#define EPS 1e-5

double* NNet::NumGrad(double* data, double* labels, unsigned N) const {
	double* delta_params = new double[this->num_params];

	double* neurons;
	double p, Jp, Jm;
	for (unsigned i = 0; i < this->num_params; i++) {
		p = this->params[i];

		this->params[i] = p + EPS;
		neurons = this->Forward(data, N);
		Jp = this->Cost(this->YHat(neurons, N), labels, N);
		delete[] neurons;

		this->params[i] = p - EPS;
		neurons = this->Forward(data, N);
		Jm = this->Cost(this->YHat(neurons, N), labels, N);
		delete[] neurons;

		delta_params[i] = 0.5 * (Jp - Jm) / EPS;

		this->params[i] = p;
	}

	return delta_params;
}