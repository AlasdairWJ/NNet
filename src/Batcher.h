#ifndef __BATCHER_H
#define __BATCHER_H

class Batcher {
private:
	double* data;
	double* labels;
	unsigned N;
	unsigned* indicies;
	unsigned data_size;
	unsigned label_size;
public:
	Batcher();
	~Batcher();
	void SetData(double* data, unsigned data_size, double* labels, unsigned label_size, unsigned N);
	void Fetch(double* batch_data, double* batch_labels, unsigned batch_N) const;
};

#endif