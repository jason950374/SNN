#pragma once
#include <vector>
#include <queue>
#include <functional> 
#include "SpkEvent.h"
#include "Layer.h"

using namespace std;

class SNN {
public:
	SNN(vector<unsigned int> neuron_nums, unsigned int input_num);
	void train(vector<vector<double>> inputs);
	void test(vector<vector<double>> inputs);
private:
	unsigned int layerNum;
	vector<vector<PreSpkEvent>> output; //save output of output layer
	                                    // [input index][event order]
	priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>> eventPool;
	vector<Layer> layers;
	void resetLayers();
	vector<PreSpkEvent> forward(vector<double> input, bool isTrain = true);
	void backward(vector<double> input);
	void applyGrade();
};