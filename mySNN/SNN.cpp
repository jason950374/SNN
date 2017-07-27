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
	allOutput = vector<vector<PreSpkEvent>>(*prev(neuron_nums.end()), vector<PreSpkEvent>());
	outputTime = vector<double>(*prev(neuron_nums.end()), 0);
	eventPool = priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>>();
}

void SNN::train(vector<vector<double>> inputs){
	allOutput.clear();
	for (auto it = inputs.begin(); it < inputs.end(); it++) {
		auto tempOut = forward(*it);
		allOutput.push_back(tempOut);
		for (auto it2 = outputTime.begin(); it2 < outputTime.end(); it2++){
			*it2 = INFINITY;
		}
		for (auto it2 = tempOut.begin(); it2 < tempOut.end(); it2++){
			if (outputTime[it2->preIndex] < it2->time){
				outputTime[it2->preIndex] = it2->time;
			}
		}
		
		backward(*it);
		applyGrade();
		resetLayers();
	}
}

void SNN::test(vector<vector<double>> inputs){
	allOutput.clear();
	for (auto it = inputs.begin(); it < inputs.end(); it++) {
		allOutput.push_back(forward(*it, false));
		resetLayers();
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
		if (newEvent.time >= 0) {
			if (newEvent.layer < layerNum) {
				vector<PostSpkEvent> newEvents = layers[newEvent.layer].preSynEvent(newEvent);
				for (auto it = newEvents.begin(); it < newEvents.end(); it++) {
					eventPool.push(*it);
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
	}
	return output;
}

void SNN::backward(vector<double> input){
	//from loss function
	unsigned int outNum = layers[layerNum - 1].getNeuronAmount();
	vector<double> gradient = vector<double>(outNum, INFINITY);
	double minTime = INFINITY;
	//TODO something wrong with this
	for (unsigned int i = 0; i < outNum; i++){
		minTime = min(minTime, outputTime[i]);
	}
	for (unsigned int i = 0; i < outNum; i++){
		gradient[i] = outputTime[i] - minTime;
	}
	vector<vector<NodeSentBP>> postNodes = vector<vector<NodeSentBP>>(outNum, vector<NodeSentBP>());
	for (unsigned int i = 0; i < outNum; i++) {
		postNodes[i].push_back(NodeSentBP(outputTime[i]));
	}
	for (int i = layerNum; i > 0; i--) {
		vector<vector<NodeReceiveBP>> preNodes;
		preNodes = layers[i - 1].getNodeReceiveBP(postNodes);
		gradient = layers[i - 1].getGrade(gradient, preNodes, postNodes);
		postNodes = layers[i - 1].getNodeSentBP(preNodes);
	}
}

void SNN::applyGrade(){
	// TODO applyGrade
}


