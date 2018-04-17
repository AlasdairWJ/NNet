#ifndef __TRAINER_H
#define __TRAINER_H

#include "NNet.h"

class Trainer {
protected:
	unsigned batch_size;
	unsigned max_epoch;
	double* data;
	double* labels;
	unsigned N;
public:
	unsigned BatchSize() const;
	unsigned MaxEpoch() const;
	void SetBatchSize(unsigned batch_size);
	void SetMaxEpoch(unsigned max_epoch);
	void SetData(double* data, double* labels, unsigned N);

	Trainer();
	virtual void Train(const NNet& nn, double* costs) const = 0;
};

class TrainSGDM : public Trainer {
protected:
	double learning_rate;
	double decay;
public:
	TrainSGDM(double learning_rate, double decay);
	void Train(const NNet& nn, double* costs) const;
};

#endif