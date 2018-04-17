#include "Trainer.h"

#include "Batcher.h"
#include <cstring>
#include <cstdlib>

Trainer::Trainer() {
	this->batch_size = 25;
	this->max_epoch = 300;
}

unsigned Trainer::BatchSize() const {
	return batch_size;
}

unsigned Trainer::MaxEpoch() const {
	return max_epoch;
}

void Trainer::SetBatchSize(unsigned batch_size) {
	this->batch_size = batch_size;
}

void Trainer::SetMaxEpoch(unsigned max_epoch) {
	this->max_epoch = max_epoch;
}

void Trainer::SetData(double* data, double* labels, unsigned N) {
	this->data = data;
	this->labels = labels;
	this->N = N;
}

TrainSGDM::TrainSGDM(double learning_rate, double decay) {
	this->learning_rate = learning_rate;
	this->decay = decay;
}

void TrainSGDM::Train(const NNet& nn, double* costs) const {
	
	double* batch_data = new double[this->batch_size * nn.DataSize()];
	double* batch_labels = new double[this->batch_size * nn.NumLabels()];
	
	Batcher batcher;
	batcher.SetData(this->data, nn.DataSize(), this->labels, nn.NumLabels(), this->N);
	
	double* params = nn.Params();
	double* vel = new double[nn.NumParams()];
	
	unsigned i;
	for (i = 0; i < nn.NumParams(); i++) {
		vel[i] = 0.0;
	}

	double* neurons;
	double* delta_params;

	for (unsigned epoch = 0; epoch < this->max_epoch; epoch++) {

		batcher.Fetch(batch_data, batch_labels, this->batch_size);

		neurons = nn.Forward(batch_data, this->batch_size);

		if (costs != NULL) {
			costs[epoch] = nn.Cost(nn.YHat(neurons, this->batch_size), batch_labels, this->batch_size);
		}

		delta_params = nn.Back(batch_data, neurons, batch_labels, this->batch_size);
		delete[] neurons;

		for (i = 0; i < nn.NumParams(); i++) {
			vel[i] = this->decay * vel[i] + learning_rate * delta_params[i];
			params[i] -= vel[i];
		}

		delete[] delta_params;
	}

	delete[] vel;
	delete[] batch_data;
	delete[] batch_labels;
}