
#include <algorithm>
#include "Synapse.h"

Synapse::Synapse(){
	delay = 0;
	weight = 1;
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
	gradeDelay += pre_Grade;
	gradeWeight += post_Grade * (exp(-leakage * time) - exp(-EPSC_degrade * time));

	return pre_Grade;
}

void Synapse::resetGrade(){
	gradeDelay = 0;
	gradeWeight = 0;
}


//get - (d(post membrane potential) / (pre fire time))
double Synapse::getGradeTemp(double time, double leakage, double EPSC_degrade) {
	//may be reusable
	return leakage * exp(-leakage * time) - EPSC_degrade * exp(-EPSC_degrade * time);
}
