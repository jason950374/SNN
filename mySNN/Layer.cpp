
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <assert.h>
#include "Layer.h"
#include "SpkEvent.h"
#include "NodeBP.h"

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
}

vector<PostSpkEvent> Layer::preSynEvent(PreSpkEvent inputEvent){
	vector<PostSpkEvent> postSpkEvents;
	for (unsigned int i = 0; i < neuronAmount; i++) {
		postSpkEvents.push_back(synapses[inputEvent.preIndex][i].preSynEvent(inputEvent));
	}
	return postSpkEvents;
}

PreSpkEvent Layer::postSynEvent(PostSpkEvent inputEvent, double endTime, bool isTrain){
	PreSpkEvent preSpkEvent;
	// get curve
	double beginTime = inputEvent.time;
	double lastTime = 0;
	if (!finishedEvent[inputEvent.postIndex].empty()) {
		//finishedEvent[inputEvent.postIndex act like stack here, need to push_back with order of time 
		lastTime = finishedEvent[inputEvent.postIndex].back().time;
	}
	leakage_coe[inputEvent.postIndex] = leakage_coe[inputEvent.postIndex] * exp(-leakage * (beginTime - lastTime));
	EPSC_degrade_coe[inputEvent.postIndex] = EPSC_degrade_coe[inputEvent.postIndex] * exp(-EPSC_degrade * (beginTime - lastTime));

	leakage_coe[inputEvent.postIndex] += inputEvent.strenth;
	EPSC_degrade_coe[inputEvent.postIndex] += inputEvent.strenth;

	// get max time
	double tmax;
	tmax = beginTime + log(leakage_coe[inputEvent.postIndex] / EPSC_degrade_coe[inputEvent.postIndex]) / (leakage - EPSC_degrade);
	endTime = min(tmax, endTime);
	double value_begin = leakage_coe[inputEvent.postIndex] *exp(-leakage*beginTime) 
		- EPSC_degrade_coe[inputEvent.postIndex] *exp(-EPSC_degrade*beginTime);
	double value_end = leakage_coe[inputEvent.postIndex] *exp(-leakage*endTime) 
		- EPSC_degrade_coe[inputEvent.postIndex] *exp(-EPSC_degrade*endTime);
	if (value_end < threshold[inputEvent.postIndex]) {
		preSpkEvent.time = -1;
		return preSpkEvent;
	}

	// get spiketime
	while ((endTime - beginTime) < 0.01 /*precision*/) {
		double slope_up;
		//Derivative at t = begin
		slope_up = leakage_coe[inputEvent.postIndex] * leakage * exp(-leakage*beginTime) -
			EPSC_degrade_coe[inputEvent.postIndex] * EPSC_degrade * exp(-EPSC_degrade*beginTime);
		double slope_below;
		//slope of the line from curve(beginTime) to curve(endTime)
		slope_below = (value_end - value_begin) / (endTime - beginTime);

		// update according to the intersection of threshold and approximate line
		beginTime += (threshold[inputEvent.postIndex] - value_begin) / slope_up;
		endTime = (threshold[inputEvent.postIndex] - value_begin) / slope_below + beginTime;
		value_begin = leakage_coe[inputEvent.postIndex] *exp(-leakage*beginTime) 
			- EPSC_degrade_coe[inputEvent.postIndex] *exp(-EPSC_degrade*beginTime);
		value_end = leakage_coe[inputEvent.postIndex] *exp(-leakage*endTime) 
			- EPSC_degrade_coe[inputEvent.postIndex] *exp(-EPSC_degrade*endTime);
	}
	preSpkEvent.preIndex = inputEvent.postIndex;
	preSpkEvent.time = beginTime;
	if (isTrain) {
		sendEvent[preSpkEvent.preIndex].push_back(preSpkEvent);
	}
	// reset neuron after spike
	leakage_coe[inputEvent.postIndex] = 0;
	EPSC_degrade_coe[inputEvent.postIndex] = 0;

	if (isTrain) {
		finishedEvent[inputEvent.postIndex].push_back(inputEvent);
		finishedEventRef[inputEvent.preIndex][inputEvent.postIndex].push_back(finishedEvent[inputEvent.postIndex].size() - 1);
	}
	else {
		if (!finishedEvent[inputEvent.postIndex].empty()) {
			finishedEvent[inputEvent.postIndex][0] = inputEvent;
		}			
		else {
			finishedEvent[inputEvent.postIndex].push_back(inputEvent);
		}
	}
	
	return preSpkEvent;
}

//vector<double> Layer::getGrade(vector<double> grade_post){
//	vector<double> grade_pre(synapses.size(), 0.0);
//	assert(grade_post.size() == neuronAmount);
//	//iterate pre index
//	for (unsigned int  i = 0; i < synapses.size(); i++) {
//		double gradeTemp = 0;
//		//iterate post index
//		for (unsigned int  j = 0; j < grade_post.size(); j++) {
//			auto it_post = sendEvent[j].begin();
//			for (auto it = finishedEventRef[i][j].begin(); it < finishedEventRef[i][j].end(); it++) {
//				if (it_post->time < finishedEvent[i][*it].time){
//					it_post = next(it_post);
//				}
//				double time = it_post->time - finishedEvent[i][*it].time;
//				gradeTemp += (synapses[i][j]).getGradeTemp(time, leakage, EPSC_degrade);
//			}
//		}
//		for (unsigned int  j = 0; j < grade_post.size(); j++) {
//			auto it_post = sendEvent[j].begin();
//			/*for (auto it = finishedEventRef[i][j].begin(); it < finishedEventRef[i][j].end(); it++) {
//				if (it_post->time < finishedEvent[i][*it].time) {
//					it_post = next(it_post);
//				}
//				double time = it_post->time - finishedEvent[i][*it].time;
//			}*/
//			// 
//			grade_pre[i] += (synapses[i][j]).get_setGrade(grade_post[j] / gradeTemp, time, leakage, EPSC_degrade);
//		}
//	}
//	return grade_pre;
//}

vector<double> Layer::getGrade(vector<double> grade_post, vector<vector<NodeReceiveBP>> preNodes, vector<vector<NodeSentBP>> postNodes){
	vector<double> grade_pre(synapses.size(), 0.0);
	assert(grade_post.size() == neuronAmount);
	assert(postNodes.size() == neuronAmount);
	for (unsigned int i = 0; i < synapses.size(); i++) {
		double gradeTemp = 0;
		//iterate post index
		for (unsigned int  j = 0; j < grade_post.size(); j++) {
			for (auto it = postNodes[j].begin(); it < postNodes[j].end(); it++){
				gradeTemp += (synapses[i][j]).getGradeTemp(it->time - preNodes[i][it->preIndex].time, 
					leakage, EPSC_degrade);
			}
		}
		for (unsigned int  j = 0; j < grade_post.size(); j++) {
			auto it_post = sendEvent[j].begin();
			for (auto it = postNodes[j].begin(); it < postNodes[j].end(); it++){
				grade_pre[i] += (synapses[i][j]).get_addGrade(grade_post[j] / gradeTemp, 
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
			postNodeBPs[i].push_back(NodeSentBP(it->time + synapses[i][it->postIndex].delay));
		}
	}
	return postNodeBPs;
}

unsigned int Layer::getNeuronAmount(){
	return neuronAmount;
}