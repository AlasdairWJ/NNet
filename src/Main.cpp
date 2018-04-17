#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "NNet.h"
#include "Trainer.h"
#include "Batcher.h"
#include "Util.h"

#define NUM_FIELDS 13
#define NUM_LABELS 3

#define NUM_SAMPLES 178
#define NUM_TRAINING_SAMPLES 150
#define NUM_TESTING_SAMPLES	28

#define LEARNING_RATE 0.1
#define DECAY 0.9
#define MAX_EPOCHS 300
#define BATCH_SIZE 25

// https://archive.ics.uci.edu/ml/datasets/Wine

int main(int argc, char* argv[]) {

	srand(time(NULL));

	double source_data[NUM_SAMPLES * NUM_FIELDS];
	LoadCSV("../data/wine.data.norm.txt", source_data, NUM_SAMPLES * NUM_FIELDS);

	double source_labels[NUM_SAMPLES * NUM_LABELS];
	LoadCSV("../data/wine.labels.yhat.txt", source_labels, NUM_SAMPLES * NUM_LABELS);

	double shuffled_data[NUM_SAMPLES * NUM_FIELDS];
	double shuffled_labels[NUM_SAMPLES * NUM_LABELS];

	Batcher batcher;
	batcher.SetData(source_data, NUM_FIELDS, source_labels, NUM_LABELS, NUM_SAMPLES);
	batcher.Fetch(shuffled_data, shuffled_labels, NUM_SAMPLES);

	double* training_data = &shuffled_data[0];
	double* training_label = &shuffled_labels[0];
	double* testing_data = &shuffled_data[NUM_TRAINING_SAMPLES * NUM_FIELDS];
	double* testing_labels = &shuffled_labels[NUM_TRAINING_SAMPLES * NUM_LABELS];

	unsigned layers[] = { NUM_FIELDS, 16, 16, 16, NUM_LABELS};
	NNet nn(layers, sizeof(layers) / sizeof(unsigned));

	printf("Number of parameters: %d\n", nn.NumParams());

	nn.Randomise();

	TrainSGDM trainer(LEARNING_RATE, DECAY);
	trainer.SetMaxEpoch(MAX_EPOCHS);
	trainer.SetBatchSize(BATCH_SIZE);
	trainer.SetData(training_data, training_label, NUM_TRAINING_SAMPLES);

	trainer.Train(nn, NULL);

	{
		puts("Testing Data:");

		double* neurons = nn.Forward(testing_data, NUM_TESTING_SAMPLES);
		double* yhat = nn.YHat(neurons, NUM_TESTING_SAMPLES);
		
		unsigned grid[NUM_LABELS][NUM_LABELS] = { 0 };

		unsigned yhat_classes[NUM_TESTING_SAMPLES];
		unsigned labels_classes[NUM_TESTING_SAMPLES];

		nn.Classify(yhat, yhat_classes, NUM_TESTING_SAMPLES);
		nn.Classify(testing_labels, labels_classes, NUM_TESTING_SAMPLES);

		for (unsigned n = 0; n < NUM_TESTING_SAMPLES; n++) {
			grid[yhat_classes[n]][labels_classes[n]]++;
		}

		unsigned i, j;
		for (j = 0; j < NUM_LABELS; j++) {
			for (i = 0; i < NUM_LABELS; i++) {
				printf("%2d ", grid[j][i]);
			}
			putchar('\n');
		}

		delete[] neurons;
	}

	putchar('\n');
	
	{
		puts("All Data:");
		
		double* neurons = nn.Forward(source_data, NUM_SAMPLES);
		double* yhat = nn.YHat(neurons, NUM_SAMPLES);
		
		unsigned grid[NUM_LABELS][NUM_LABELS] = { 0 };

		unsigned yhat_classes[NUM_SAMPLES];
		unsigned labels_classes[NUM_SAMPLES];

		nn.Classify(yhat, yhat_classes, NUM_SAMPLES);
		nn.Classify(source_labels, labels_classes, NUM_SAMPLES);

		for (unsigned n = 0; n < NUM_SAMPLES; n++) {
			grid[yhat_classes[n]][labels_classes[n]]++;
		}

		unsigned i, j;
		for (j = 0; j < NUM_LABELS; j++) {
			for (i = 0; i < NUM_LABELS; i++) {
				printf("%2d ", grid[j][i]);
			}
			putchar('\n');
		}

		delete[] neurons;
	}

	return 0;
}