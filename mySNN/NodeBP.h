#pragma once

class NodeSentBP {
public:
	NodeSentBP(double time, unsigned int preIndex = 0) {
		this->time = time;
		this->preIndex = preIndex;
	}
	double time;
	unsigned int preIndex;
};

class NodeReceiveBP {
public:
	NodeReceiveBP(double time, unsigned int postIndex) {
		this->time = time;
		this->postIndex = postIndex;
	}
	double time;
	unsigned int postIndex;
};

