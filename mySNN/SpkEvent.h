#pragma once


class PostSpkEvent {
public:
	double time;
	//location
	unsigned int layer; //belong to post-neuron
	unsigned int postIndex;
	unsigned int preIndex; //for backward
	double strenth;
};

bool operator <(const PostSpkEvent& e1, const PostSpkEvent& e2);
bool operator >(const PostSpkEvent& e1, const PostSpkEvent& e2);

//temparary event
//turn to PostSpkEvent after Synapse preSynEvent
class PreSpkEvent {
public:
	double time;
	//location
	unsigned int layer; //belong to post-neuron
	unsigned int preIndex;
};
