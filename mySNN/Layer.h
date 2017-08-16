#pragma once
#include <vector>
#include "SpkEvent.h"
#include "Synapse.h"
#include "NodeBP.h"

using namespace std;

class Layer {
public:
	/******************************** member ****************************************/
	vector<vector<Synapse>> synapses; //from previous layer, means post-neuron is in current layer
									  //[pre][post]

	/******************************** method ****************************************/
	Layer(unsigned int neuronAmount, unsigned int prLayerNeuronAmount);
	void resetLayer();
	vector<PostSpkEvent> preSynEvent(PreSpkEvent inputEvent);
	PreSpkEvent postSynEvent(PostSpkEvent inputEvent, double endTime, bool isTrain = true, bool isStall = true);
	vector<double> getGrade(vector<double> grade_pre, vector<vector<NodeReceiveBP>> preNodes, vector<vector<NodeSentBP>> postNodes);
	
	//get NodeReceiveBP-NodeSentBP relation acording to relative time of sent and receive spike
	vector<vector<NodeReceiveBP>> getNodeReceiveBP(vector<vector<NodeSentBP>> postNodeBPs);  
	
	//add delay, BP for previous layer
	vector<vector<NodeSentBP>> getNodeSentBP(vector<vector<NodeReceiveBP>> preNodeBPs);

	void applyGrade(double learningRate);
	void balance();
	unsigned int getNeuronAmount();
	bool getStall(unsigned int index);
	PreSpkEvent Layer::sovleStall(int index, double endTime, bool isTrain);
private:
	/******************************** member ****************************************/
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

	// if second event is not same index, stall the neuron and set this flag true
	vector<bool> stalls;
	vector<double> lastUpdateTime;

	/******************************** method ****************************************/
	double getSpiketime(double beginTime, double endTime, double value_begin, double value_end, int index, double precision);
};
