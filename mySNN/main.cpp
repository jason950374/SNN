#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <random>
#include "SNN.h"

//#define NDEBUG

using namespace std;

void normalize(double feature[][150]);
void normalize(double feature[][150], double min_num, double max_num);
void valueToDelay(double feature[][150], double min_num, double max_num);
void shuffle_both(double feature[][150], unsigned char * label);


int main(void) {
	ifstream input("bezdekIris.data");
	double feature[4][150];
	unsigned char label[150];
	char s[20];
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			input >> feature[j][i];
		}
		input >> s;
		if (strcmp(s, "Iris-setosa") == 0){
			label[i] = 0;
		}
		else if (strcmp(s, "Iris-versicolor") == 0){
			label[i] = 1;
		}
		else if (strcmp(s, "Iris-virginica") == 0){
			label[i] = 2;
		}
		else {
			label[i] = 100;
		}
	}
	input.close();

	normalize(feature, 0, 50);
	valueToDelay(feature, 0, 50); //larger value have smaller delay
	shuffle_both(feature, label);
	
	vector<unsigned int> neuron_nums = vector<unsigned int>({3, 3});

	SNN snn = SNN(neuron_nums, 4);

	//snn.train();
	vector<vector<double>> feature_vector(150, vector<double>(4, 0));
	for (unsigned int i = 0; i < 150; i++) {
		for (unsigned int j = 0; j < 4; j++)
			feature_vector[i][j] = feature[j][i];
	}
	snn.test(feature_vector);

	return 0;
}

void normalize(double feature[][150]) {
	double max_feature[4] = { 0 };
	double min_feature[4] = { 1024, 1024, 1024, 1024 };
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			max_feature[j] = max(max_feature[j], feature[j][i]);
			min_feature[j] = min(min_feature[j], feature[j][i]);
		}
	}
	double dev[4];
	for (int j = 0; j < 4; j++) {
		dev[j] = max_feature[j] - min_feature[j];
	}
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			feature[j][i] = feature[j][i] - min_feature[j];
			feature[j][i] = feature [j][i] / dev[j];
		}
	}
}

void normalize(double feature[][150], double min_num , double max_num) {
	double max_feature[4] = { 0 };
	double min_feature[4] = { 1024, 1024, 1024, 1024 };
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			max_feature[j] = max(max_feature[j], feature[j][i]);
			min_feature[j] = min(min_feature[j], feature[j][i]);
		}
	}
	double dev[4];
	for (int j = 0; j < 4; j++) {
		dev[j] = max_feature[j] - min_feature[j];
	}
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			feature[j][i] = feature[j][i] - min_feature[j];
			feature[j][i] = feature[j][i] / dev[j];
			feature[j][i] = feature[j][i] * (max_num - min_num);
			feature[j][i] = feature[j][i] + min_num;
		}
	}
}

void valueToDelay(double feature[][150], double min_num, double max_num){
	for (int i = 0; i < 150; i++) {
		for (int j = 0; j < 4; j++) {
			feature[j][i] = max_num - feature[j][i];
		}
	}
}

void shuffle_both(double feature[][150], unsigned char * label) {
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine g = default_random_engine(seed);
	for (int i = 149; i > 0; --i) {
		uniform_int_distribution<int> d(0, i);
		int temp = d(g);
		for (int j = 0; j < 4; j++) {
			swap(feature[j][i], feature[j][temp]);
		}
		swap(label[i], label[temp]);
	}
}








