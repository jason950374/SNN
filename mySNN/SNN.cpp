#include "SNN.h"
#include "SNN.h"
#include <limits>
#include <algorithm>
#include <assert.h>
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
	outputTime = vector<double>(*prev(neuron_nums.end()), INFINITY);
	eventPool = EventPool();
}

void SNN::train(vector<vector<double>> inputs, vector<unsigned char> label, double learningRate){
	allOutput.clear();
	int i = 0;
	for (auto it = inputs.begin(); it < inputs.end(); it++, i++) {
		resetLayers();
		auto tempOut = forward(*it);
		allOutput.push_back(tempOut);
		for (auto it2 = outputTime.begin(); it2 < outputTime.end(); it2++){
			*it2 = INFINITY;
		}
		for (auto it2 = tempOut.begin(); it2 < tempOut.end(); it2++){
			outputTime[it2->preIndex] = min(outputTime[it2->preIndex], it2->time);
		}
		backward(*it, label[i]);
		applyGrade(learningRate);
		//balance();
	}
}

void SNN::test(vector<vector<double>> inputs){
	allOutput.clear();
	for (auto it = inputs.begin(); it < inputs.end(); it++) {
		resetLayers();
		allOutput.push_back(forward(*it, false));
	}
}

unsigned int SNN::getOutput(unsigned int index){
	if(!allOutput[index].empty()) 
		return allOutput[index][0].preIndex;
	return (--layers.end())->getNeuronAmount();
}

void SNN::resetLayers(){
	//clear remain event (inference will end when get the result)
	eventPool.reset();
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
		for (auto it2 = newEvents.begin(); it2 < newEvents.end(); it2++) {
			int err = eventPool.push(*it2);
			assert(err == 0);
		}
	}
	while (!eventPool.empty()) {
		Event & curEvent = eventPool.top();
		eventPool.pop();
		PostSpkEvent secondEvent;
		if (!eventPool.emptyPost()) {
			secondEvent = eventPool.topPost();
		}
		else {
			secondEvent.time = numeric_limits<double>::max();
		}
		Event & newEvent = layers[curEvent.layer].postSynEvent(curEvent, secondEvent, isTrain);

		if (newEvent.time >= 0) {
			// not output
			if (newEvent.layer < layerNum) {
				if (newEvent.ID == PreSpkEventID){
					vector<PostSpkEvent> newEvents = layers[newEvent.layer].preSynEvent(*dynamic_cast<PreSpkEvent*> (&newEvent));
					for (auto it = newEvents.begin(); it < newEvents.end(); it++) {
						assert(it->time >= 0);
						int err = eventPool.push(*it);
						assert(err == 0);
					}
				}
				else if (newEvent.ID == CheckEventID) {
					int err = eventPool.push(*dynamic_cast<CheckEvent*> (&newEvent));
					assert(err == 0);
				}
			}
			else {
				output.push_back(*dynamic_cast<PreSpkEvent*> (&newEvent));
				if (!isTrain) {
					//inference, just need first spike
					return output;
				}
			}
		}
		delete &newEvent;
		delete &curEvent;
	}
	return output;
}

void SNN::backward(vector<double> input, unsigned int label){
	//from loss function
	unsigned int outNum = layers[layerNum - 1].getNeuronAmount();
	vector<double> gradient = vector<double>(outNum, INFINITY);
	double minTime = INFINITY;
	double maxTime = 0;
	for (unsigned int i = 0; i < outNum; i++){
		minTime = min(minTime, outputTime[i]);
	}
	for (unsigned int i = 0; i < outNum; i++) {
		maxTime = max(maxTime, outputTime[i]);
	}
	for (unsigned int i = 0; i < outNum; i++){
		if (i == label)
			if (!isinf(minTime))
				gradient[i] = min(outputTime[i] - minTime, 10.0); //clip gradient for stability
				//gradient[i] = max(minTime - outputTime[i], -10.0); //clip gradient for stability
			else
				gradient[i] = 10;
		else
			if (!isinf(outputTime[i]))
				//gradient[i] = min(maxTime - outputTime[i], 10.0); //clip gradient for stability
				gradient[i] = max(outputTime[i] - maxTime, -10.0); //clip gradient for stability
			else
				gradient[i] = 0;
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

void SNN::applyGrade(double learningRate){
	for (auto it = layers.begin(); it < layers.end(); it++) {
		it->applyGrade(learningRate);
	}
}

void SNN::balance() {
	for (auto it = layers.begin(); it < layers.end(); it++) {
		it->balance();
	}
}

SNN::EventPool::EventPool(){
	postEventPool = priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>>();
	checkEventPool = priority_queue<CheckEvent, vector<CheckEvent>, greater<CheckEvent>>();
}

Event & SNN::EventPool::top() {
	PostSpkEvent & postEvent = *(new PostSpkEvent());
	CheckEvent & checkEvent = *(new CheckEvent());;
	if (postEventPool.empty()) {
		checkEvent = checkEventPool.top();
		delete &postEvent;
		return checkEvent;
	}
	postEvent = postEventPool.top();
	if (checkEventPool.empty()) {
		delete &checkEvent;
		return postEvent;
	}
	checkEvent = checkEventPool.top();
	if (checkEvent < postEvent) {
		delete &postEvent;
		return checkEvent;
	}
	delete &checkEvent;
	return postEvent;
}

void SNN::EventPool::pop() {
	if (postEventPool.empty()) {
		checkEventPool.pop();
		return;
	}
	if (checkEventPool.empty()) {
		postEventPool.pop();
		return;
	}

	PostSpkEvent postEvent = postEventPool.top();
	CheckEvent checkEvent = checkEventPool.top();
	if (checkEvent < postEvent) {
		checkEventPool.pop();
		return;
	}
	postEventPool.pop();
	return;
}

PostSpkEvent SNN::EventPool::topPost(){
	return postEventPool.top();
}

void SNN::EventPool::reset(){
	postEventPool = priority_queue<PostSpkEvent, vector<PostSpkEvent>, greater<PostSpkEvent>>();
	checkEventPool = priority_queue<CheckEvent, vector<CheckEvent>, greater<CheckEvent>>();
}

unsigned char SNN::EventPool::push(Event &e){
	if (e.ID == PostSpkEventID) {
		postEventPool.push(*dynamic_cast<PostSpkEvent *> (&e));
	}
	else if(e.ID == CheckEventID){
		checkEventPool.push(*dynamic_cast<CheckEvent *> (&e));
	}
	else {
		return -1;
	}
	return 0;
}

bool SNN::EventPool::empty(){
	return postEventPool.empty() && checkEventPool.empty();
}

bool SNN::EventPool::emptyPost(){
	return postEventPool.empty();
}
