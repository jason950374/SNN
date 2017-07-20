#include "SNN.h"
#include "SNN.h"
#include <limits>
#include <vector>

using namespace std;

SNN::SNN(vector<unsigned int> neuron_nums, unsigned int input_num){
	this->layers = vector<Layer>();
	this->layers.push_back(Layer(neuron_nums[0], input_num));
	for (auto it = neuron_nums.begin() + 1; it < neuron_nums.end(); it++) {
		this->layers.push_back(Layer(*it, *(it-1)));
	}
	layerNum = layers.size();
	output = vector<vector<PreSpkEvent>>();
	eventPool = priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>>();
}

void SNN::train(vector<vector<double>> inputs){
	output.clear();
	for (auto it = inputs.begin(); it < inputs.end(); it++) {
		output.push_back(forward(*it));
		resetLayers();
		backward(*it);
		applyGrade();
	}
}

void SNN::test(vector<vector<double>> inputs){
	output.clear();
	for (auto it = inputs.begin(); it < inputs.end(); it++) {
		output.push_back(forward(*it, false));
	}
}

void SNN::resetLayers(){
	for (auto it = layers.begin(); it < layers.end(); it++) {
		it->resetLayer();
	}
}

vector<PreSpkEvent> SNN::forward(vector<double> input, bool isTrain){
	vector<PreSpkEvent> output;
	unsigned int i = 0;
	for (auto it = input.begin(); it < input.end(); it++, i++) {
		PreSpkEvent newEvent;
		newEvent.layer = 0;
		newEvent.preIndex = i;
		newEvent.time = *it;
		vector<PostSpkEvent> newEvents = layers[0].preSynEvent(newEvent);
		for (auto it = newEvents.begin(); it < newEvents.end(); it++) {
			eventPool.push(*it);
		}
	}
	while (!eventPool.empty()) {
		PostSpkEvent curEvent = eventPool.top();
		eventPool.pop();
		PostSpkEvent secondEvent;
		if (!eventPool.empty()) {
			secondEvent = eventPool.top();
		}
		else {
			secondEvent.time = numeric_limits<double>::max();
		}
		PreSpkEvent newEvent = layers[curEvent.layer].postSynEvent(curEvent, secondEvent.time, isTrain);
		newEvent.layer = curEvent.layer + 1;
		// not output
		if (newEvent.layer < layerNum) {
			if (newEvent.time >= 0) {
				vector<PostSpkEvent> newEvents = layers[newEvent.layer].preSynEvent(newEvent);
				for (auto it = newEvents.begin(); it < newEvents.end(); it++) {
					eventPool.push(*it);
				}
			}
		}
		else {
			output.push_back(newEvent);
			if (!isTrain) {
				//inference, just need first spike
				return output;
			}
		}
	}
	return output;
}

void SNN::backward(vector<double> input){
	//gradient; //from loss function
	for (auto it = prev(layers.end()); it >= layers.begin(); --it) {
		//gradient = it->getGrade(gradient);
	}
	// TODO backward
}

void SNN::applyGrade(){
	// TODO applyGrade
}


