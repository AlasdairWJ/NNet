#include "Batcher.h"

#include <cstdlib>
#include <cstring>

Batcher::Batcher() {
	this->indicies = NULL;
}

Batcher::~Batcher() {
	if (this->indicies) {
		delete[] this->indicies;
	}
}

void Batcher::SetData(double* data, unsigned data_size, double* labels, unsigned label_size, unsigned N) {
	this->data = data;
	this->labels = labels;
	this->N = N;

	if (this->indicies) {
		delete[] this->indicies;
	}

	this->indicies = new unsigned[N];
	for (unsigned i = 0; i < N; i++) {
		this->indicies[i] = i;
	}

	this->data_size = data_size;
	this->label_size = label_size;
}

void Batcher::Fetch(double* batch_data, double* batch_labels, unsigned batch_N) const {
	unsigned i, j, index, ni, nj;
	for (i = 0; i < batch_N; i++) {
		index = i + rand() % (this->N - i);
		j = this->indicies[index];
		this->indicies[index] = this->indicies[i];
		this->indicies[i] = j;

		ni = i * this->data_size;
		nj = j * this->data_size; 
		std::memcpy(&batch_data[ni], &this->data[nj], this->data_size * sizeof(double));

		ni = i * this->label_size;
		nj = j * this->label_size; 
		std::memcpy(&batch_labels[ni], &this->labels[nj], this->label_size * sizeof(double));
	}
}

