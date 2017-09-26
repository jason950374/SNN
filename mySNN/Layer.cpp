
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <assert.h>
#include <limits>
#include "Layer.h"
#include "SpkEvent.h"
#include "NodeBP.h"

#define EPSILON 0.00000000001
#define BLANCE_COE 1.1

using namespace std;

Layer::Layer(unsigned int neuronAmount, unsigned int prLayerNeuronAmount){
	this->neuronAmount = neuronAmount;
	this->synapses = vector<vector<Synapse>>(prLayerNeuronAmount,
		vector<Synapse>(neuronAmount, Synapse()));
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine g = default_random_engine(seed);
	for (auto it = synapses.begin(); it < synapses.end(); it++) {
		for (auto it2 = it->begin(); it2 < it->end(); it2++) {
			uniform_real_distribution<double> distribution(0.0, 10.0);
			double temp = distribution(g);
			it2->delay = temp;
		}
	}

	leakage = 0.1;
	EPSC_degrade = 0.3;
	spikeCnt = vector<unsigned int>(neuronAmount, 0);
	threshold = vector<double>(neuronAmount, 0.5);
	finishedEvent = vector<vector<PostSpkEvent>>(neuronAmount, vector<PostSpkEvent>());
	finishedEventRef = vector<vector<vector<unsigned int>>>(prLayerNeuronAmount, 
		vector<vector<unsigned int>>(neuronAmount, vector<unsigned int>()));
	leakage_coe = vector<double>(neuronAmount, 0);
	EPSC_degrade_coe = vector<double>(neuronAmount, 0);

	sendEvent = vector<vector<PreSpkEvent>>(neuronAmount, vector<PreSpkEvent>());
	lastUpdateTime = vector<double>(neuronAmount, 0);
}

void Layer::resetLayer(){
	spikeCnt = vector<unsigned int>(neuronAmount, 0);
	assert(leakage_coe.size() == EPSC_degrade_coe.size());
	for (unsigned int i = 0; i < leakage_coe.size(); i++) {
		leakage_coe[i] = 0;
		EPSC_degrade_coe[i] = 0;
		for (auto it = sendEvent.begin(); it < sendEvent.end(); it++){
			it->clear();
		}
		for (auto it = finishedEvent.begin(); it < finishedEvent.end(); it++){
			it->clear();
		}
		for (auto it = finishedEventRef.begin(); it < finishedEventRef.end(); it++) {
			for (auto it2 = it->begin(); it2 < it->end(); it2++) {
				it2->clear();
			}
		}
		for (unsigned int j = 0; j < synapses.size(); j++) {
			synapses[j][i].resetGrade();
		}
	}
	lastUpdateTime = vector<double>(neuronAmount, 0);
}

vector<PostSpkEvent> Layer::preSynEvent(PreSpkEvent inputEvent){
	vector<PostSpkEvent> postSpkEvents;
	if (inputEvent.pseudo) {
		PostSpkEvent temp;
		temp.layer    = inputEvent.layer;
		temp.time = inputEvent.time;
		temp.strenth = 0;
		temp.preIndex = UINT_MAX;
		temp.postIndex = inputEvent.preIndex;
		postSpkEvents.push_back(temp);
	}
	else {
		for (unsigned int i = 0; i < neuronAmount; i++) {
			PostSpkEvent temp = synapses[inputEvent.preIndex][i].preSynEvent(inputEvent);
			temp.postIndex = i;
			postSpkEvents.push_back(temp);
		}
	}
	return postSpkEvents;
}

PreSpkEvent Layer::postSynEvent(PostSpkEvent inputEvent, PostSpkEvent secondEvent, bool isTrain){
	PreSpkEvent preSpkEvent;
	// get curve
	double beginTime = inputEvent.time;
	assert(inputEvent.time >= 0);

	if (isTrain) {
		if (inputEvent.preIndex != UINT_MAX) {
			finishedEvent[inputEvent.postIndex].push_back(inputEvent);
			finishedEventRef[inputEvent.preIndex][inputEvent.postIndex].push_back(finishedEvent[inputEvent.postIndex].size() - 1);
		}
	}
	assert(beginTime >= lastUpdateTime[inputEvent.postIndex]);

	leakage_coe[inputEvent.postIndex] = leakage_coe[inputEvent.postIndex] * exp(-leakage * (beginTime - lastUpdateTime[inputEvent.postIndex]));
	EPSC_degrade_coe[inputEvent.postIndex] = EPSC_degrade_coe[inputEvent.postIndex] * exp(-EPSC_degrade * (beginTime - lastUpdateTime[inputEvent.postIndex]));

	leakage_coe[inputEvent.postIndex] += inputEvent.strenth;
	EPSC_degrade_coe[inputEvent.postIndex] += inputEvent.strenth;

	lastUpdateTime[inputEvent.postIndex] = beginTime;

	// get max time
	double tmax;
	tmax = beginTime + log(EPSC_degrade / leakage)
		/ (EPSC_degrade * EPSC_degrade_coe[inputEvent.postIndex] - (leakage * leakage_coe[inputEvent.postIndex]));
	double endTime = min(tmax, secondEvent.time);
	assert(endTime >= beginTime);

	double value_begin = leakage_coe[inputEvent.postIndex] - EPSC_degrade_coe[inputEvent.postIndex];
	double value_end = leakage_coe[inputEvent.postIndex] *exp(-leakage*(endTime - beginTime))
		- EPSC_degrade_coe[inputEvent.postIndex] *exp(-EPSC_degrade*(endTime - beginTime));
	assert(value_begin < threshold[inputEvent.postIndex]);

	if (value_end < threshold[inputEvent.postIndex]) {
		if (endTime < tmax && (inputEvent.postIndex != secondEvent.postIndex) && secondEvent.preIndex != UINT_MAX) {
			preSpkEvent.pseudo = true;
			preSpkEvent.time = endTime;
			preSpkEvent.preIndex = inputEvent.postIndex;
			preSpkEvent.layer = inputEvent.layer;
		}
		else {
			preSpkEvent.time = -1;
		}
		return preSpkEvent;
	}
	double spikeTime = getSpiketime(beginTime, endTime, value_begin, value_end, inputEvent.postIndex, 0.001);
	preSpkEvent.preIndex = inputEvent.postIndex;
	preSpkEvent.time = spikeTime;
	if (isTrain) {
		sendEvent[preSpkEvent.preIndex].push_back(preSpkEvent);
	}
	// reset neuron after spike
	leakage_coe[inputEvent.postIndex] = 0;
	EPSC_degrade_coe[inputEvent.postIndex] = 0;

	spikeCnt[preSpkEvent.preIndex]++;
	assert(preSpkEvent.time >= inputEvent.time);
	preSpkEvent.layer = inputEvent.layer + 1;
	return preSpkEvent;
}

vector<double> Layer::getGrade(vector<double> grade_post, vector<vector<NodeReceiveBP>> preNodes, vector<vector<NodeSentBP>> postNodes){
	vector<double> grade_pre(synapses.size(), 0.0);
	assert(grade_post.size() == neuronAmount);
	assert(postNodes.size() == neuronAmount);
	for (unsigned int i = 0; i < synapses.size(); i++) {
		double gradeTemp = 0;
		//iterate post index
		for (unsigned int  j = 0; j < grade_post.size(); j++) {
			for (auto it = postNodes[j].begin(); it < postNodes[j].end(); it++){
				if (!isinf(preNodes[i][it->preIndex].time))
					gradeTemp += (synapses[i][j]).getGradeTemp(it->time - preNodes[i][it->preIndex].time, 
						leakage, EPSC_degrade);
			}
		}
		for (unsigned int  j = 0; j < grade_post.size(); j++) {
			assert(!isinf(grade_post[j]));
			for (auto it = postNodes[j].begin(); it < postNodes[j].end(); it++){
				if(!isinf(it->time) && !isinf(preNodes[i][it->preIndex].time))
					grade_pre[i] += (synapses[i][j]).get_addGrade(grade_post[j] / (gradeTemp + EPSILON),
						it->time - preNodes[i][it->preIndex].time, leakage, EPSC_degrade);
			}
		}
	}
	return grade_pre;
}

vector<vector<NodeReceiveBP>> Layer::getNodeReceiveBP(vector<vector<NodeSentBP>> postNodeBPs){
	vector<vector<NodeReceiveBP>> preNodeBPs = vector<vector<NodeReceiveBP>>(synapses.size(), vector<NodeReceiveBP>());
	assert(postNodeBPs.size() == neuronAmount);
	for (unsigned int i = 0; i < neuronAmount; i++) {
		for (auto it = postNodeBPs[i].begin(); it < postNodeBPs[i].end(); it++) {
			for (unsigned int j = 0; j < synapses.size(); j++) {
				double time = INFINITY;
				for (auto it2 = finishedEventRef[j][i].begin(); it2 < finishedEventRef[j][i].end(); it2++) {
					// TODO delta tMax???
					if (finishedEvent[i][*it2].time < it->time) {
						time = finishedEvent[i][*it2].time;
					}
				}
				it->preIndex = preNodeBPs[j].size();
				assert(time > 0);
				preNodeBPs[j].push_back(NodeReceiveBP(time, i));
			}
		}
	}
	return preNodeBPs;
}

vector<vector<NodeSentBP>> Layer::getNodeSentBP(vector<vector<NodeReceiveBP>> preNodeBPs){
	vector<vector<NodeSentBP>> postNodeBPs = vector<vector<NodeSentBP>>(preNodeBPs.size(), vector<NodeSentBP>());
	for (unsigned int i = 0; i < synapses.size(); i++) {
		for (auto it = preNodeBPs[i].begin(); it < preNodeBPs[i].end(); it++) {
			assert(it->time > 0);
			postNodeBPs[i].push_back(NodeSentBP(it->time + synapses[i][it->postIndex].delay));
		}
	}
	return postNodeBPs;
}

void Layer::applyGrade(double learningRate){
	for (unsigned int i = 0; i < synapses.size(); i++) {
		for (unsigned int j = 0; j < neuronAmount; j++) {
			synapses[i][j].applyGrade(learningRate);
		}
	}
}

void Layer::balance(){
	for (unsigned int i = 0; i < neuronAmount; i++) {
		for (unsigned int j = 0; j < synapses.size(); j++) {
			if(spikeCnt[i] == 0)
				synapses[j][i].weight = synapses[j][i].weight * BLANCE_COE;
			else if(spikeCnt[i] > 2)
				synapses[j][i].weight = synapses[j][i].weight / BLANCE_COE;
		}
	}
}

unsigned int Layer::getNeuronAmount(){
	return neuronAmount;
}

double Layer::getSpiketime(double beginTime, double endTime, double value_begin, double value_end, int index, double precision){
	double lowTime = beginTime;
	double highTime = endTime;
	assert(endTime > beginTime);
	while (abs(highTime - lowTime) > precision) {
		double slope_up;
		//Derivative at t = begin
		slope_up = EPSC_degrade_coe[index] * EPSC_degrade * exp(-EPSC_degrade*(lowTime - beginTime))
			- leakage_coe[index] * leakage * exp(-leakage*(lowTime - beginTime));
		double slope_below;
		//slope of the line from curve(beginTime) to curve(endTime)
		slope_below = (value_end - value_begin) / (highTime - lowTime);

		// update according to the intersection of threshold and approximate line
		lowTime += (threshold[index] - value_begin) / slope_up;
		highTime = (threshold[index] - value_begin) / slope_below + lowTime;
		value_begin = leakage_coe[index] * exp(-leakage*(lowTime - beginTime))
			- EPSC_degrade_coe[index] * exp(-EPSC_degrade*(lowTime - beginTime));
		value_end = leakage_coe[index] * exp(-leakage*(highTime - beginTime))
			- EPSC_degrade_coe[index] * exp(-EPSC_degrade*(highTime - beginTime));
	}
	assert(lowTime > beginTime);
	return lowTime;
}

