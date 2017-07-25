#pragma once

class NodeBP {
public:
	NodeBP(double time, unsigned int preIndex) {
		this->time = time;
		this->preIndex = preIndex;
	}
	double time;
	unsigned int preIndex;
private:

};


