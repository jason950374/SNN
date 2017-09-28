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
	void train(vector<vector<double>> inputs, vector<unsigned char> label, double learningRate);
	void test(vector<vector<double>> inputs);
	unsigned int getOutput(unsigned int index);
private:
	unsigned int layerNum;
	vector<vector<PreSpkEvent>> allOutput; //save output of output layer
	                                       // [input index][event order]
	vector<double> outputTime;             //for training
	vector<Layer> layers;
	void resetLayers();
	vector<PreSpkEvent> forward(vector<double> input, bool isTrain = true);
	void backward(vector<double> input, unsigned int label);
	void applyGrade(double learningRate);
	void balance();

	class EventPool {
	public:
		EventPool();
		Event & top(); //pass by reference, be carefull before doing pop
		PostSpkEvent topPost();
		void pop();
		void reset();
		unsigned char push(Event &e);
		bool empty();
		bool emptyPost();
	private:
		priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>> postEventPool;
		priority_queue<CheckEvent, vector<CheckEvent>, greater<CheckEvent>> checkEventPool;
	};

	EventPool eventPool;
};