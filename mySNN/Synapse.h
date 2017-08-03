#pragma once
#include "SpkEvent.h"

class Synapse {
public:
	Synapse();
	Synapse(double delay, double weight);
	double delay;
	double weight;
	
	PostSpkEvent preSynEvent(PreSpkEvent e); //add delay

	/***********************************************************
	get gradient for back propagation 
	 & add gradient for delay\weight of this Synapse at same time
	post_Grade is d(post fire time) / d(post membrane potential)
	************************************************************/
	double get_addGrade(double post_Grade, double time, double leakage, double EPSC_degrade);
	void resetGrade(); 
	void applyGrade(double learningRate);

	//get - (d(post membrane potential) / (pre fire time))
	double getGradeTemp(double time, double leakage, double EPSC_degrade);
private:
	double gradeDelay;
	double gradeWeight;
};



