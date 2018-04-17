#ifndef __NNET_H
#define __NNET_H

class NNet {
private:
	unsigned data_size; // number of neurons in the input layer
	unsigned num_labels; // number of neurons in the output layer 

	unsigned num_neurons; // total number of neurons in all layers
	
	unsigned num_layers; // total number of layers including input and output
	unsigned* layer_sizes; // number of neurons in a given layer

	double* weights;  unsigned num_weights;
	double* biases;   unsigned num_biases;
	double* params;   unsigned num_params;

public:
	double* Params() const;
	unsigned NumParams() const;

	unsigned DataSize() const;
	unsigned NumLabels() const;

	NNet(unsigned* layer_sizes, unsigned num_layers);
	~NNet();

	void Randomise() const;

	void Save(const char* filaname) const;
	void Load(const char* filaname) const;

	// Selects most likely class from probability distributions
	void Classify(double* yhat, unsigned* classes, unsigned N) const;

	// Jump to final layer of neuron list
	double* YHat(double* neurons, unsigned N) const;

	// Cost of yhat when labels is the true value
	double Cost(double* yhat, double* labels, unsigned N) const;

	// Forward propagate data set
	double* Forward(double* input, unsigned N) const;

	// Backpropagate data set, return gradient of paramaters
	double* Back(double* input, double* neurons, double* labels, unsigned N) const;

	void PrintNeurons(double* input, double* neurons, unsigned N) const;
	void PrintParams(double* params) const;

	// Numerical gradient checker, return gradient of paramaters
	double* NumGrad(double* yhat, double* labels, unsigned N) const;
};

#endif