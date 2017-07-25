#pragma once
#include <vector>
#include "SpkEvent.h"
#include "Synapse.h"
#include "NodeBP.h"

using namespace std;

class Layer {
public:
	Layer(unsigned int neuronAmount, unsigned int prLayerNeuronAmount);
	void resetLayer();
	vector<vector<Synapse>> synapses; //from previous layer, means post-neuron is in current layer
									  //[pre][post]
	vector<PostSpkEvent> preSynEvent(PreSpkEvent inputEvent);
	PreSpkEvent postSynEvent(PostSpkEvent inputEvent, double endTime, bool isTrain = true);
	vector<double> getGrade(vector<double> grade_pre, vector<vector<NodeBP>> preNodes, vector<vector<NodeBP>> postNodes);
	vector<vector<NodeBP>> getPreNodeBPs(vector<vector<NodeBP>> postNodeBPs);
	unsigned int getNeuronAmount();
private:
	unsigned int neuronAmount;
	double leakage;
	double EPSC_degrade;
	vector<vector<PreSpkEvent>> sendEvent;         // just for training, need to compare with time of finishedEvent
	vector<vector<PostSpkEvent>> finishedEvent;    // for inference, just need last event ,for training, need all event
												   // use the order of time
	vector<vector<vector<unsigned int>>> finishedEventRef; // use the order of pre-neuron index, the array save the index
														   // [pre][post]
	vector<double> leakage_coe;
	vector<double> EPSC_degrade_coe;
	vector<unsigned int> spikeCnt;
	vector<double> threshold;
	vector<double> gradeThreshold;
};
