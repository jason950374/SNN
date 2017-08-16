
#include <algorithm>
#include <assert.h>
#include <algorithm>
#include "Synapse.h"

using namespace std;

Synapse::Synapse(){
	delay = 0;
	weight = 10;
}

Synapse::Synapse(double delay, double weight){
	this->delay = delay;
	this->weight = weight;
}

PostSpkEvent Synapse::preSynEvent(PreSpkEvent e){
	PostSpkEvent newEvent = PostSpkEvent();
	newEvent.layer = e.layer;
	newEvent.time = e.time + delay;
	newEvent.strenth = weight;
	newEvent.preIndex = e.preIndex;
	return newEvent;
}


/***********************************************************
get gradient for back propagation (d(Loss) / d(pre fire time))
& add gradient for delay\weight of this Synapse at same time
post_Grade is d(post fire time) / d(post membrane potential)
************************************************************/
double Synapse::get_addGrade(double post_Grade, double time, double leakage, double EPSC_degrade){
	double pre_Grade = post_Grade * (leakage * exp(-leakage * time) - EPSC_degrade * exp(-EPSC_degrade * time));
	assert(!isnan(post_Grade));
	assert(!isnan(pre_Grade));
	assert(!isinf(pre_Grade));
	gradeDelay += pre_Grade;
	gradeWeight += post_Grade * (exp(-leakage * time) - exp(-EPSC_degrade * time));
	assert(!isnan(gradeDelay));
	assert(!isnan(gradeWeight));
	return pre_Grade;
}

void Synapse::resetGrade(){
	gradeDelay = 0;
	gradeWeight = 0;
}

void Synapse::applyGrade(double learningRate){
	assert(!isnan(gradeDelay));
	assert(!isnan(gradeWeight));
	//TODO diff learningRate for delay & weight ??
	delay = delay - 0.01*learningRate * gradeDelay;
	weight = weight - learningRate * gradeWeight;
	//clip
	delay = max(delay, 0.0);
	weight = min(max(weight, -10.0), 10.0);
}


//get - (d(post membrane potential) / (pre fire time))
double Synapse::getGradeTemp(double time, double leakage, double EPSC_degrade) {
	//may be reusable
	assert(!isnan(time));
	return leakage * exp(-leakage * time) - EPSC_degrade * exp(-EPSC_degrade * time);
}
