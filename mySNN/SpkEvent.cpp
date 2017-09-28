#include "SpkEvent.h"
#include <limits>

bool Event::operator<( const Event & e2) const {
	return this->time < e2.time;
}

bool Event::operator>( const Event & e2) const {
	return this->time > e2.time;
}

void Event::operator=(const Event & e2){
	time = e2.time;
	layer = e2.layer;
}

void PostSpkEvent::operator=(const PostSpkEvent & e2){
	__super::operator=(e2);
	postIndex = e2.postIndex;
	preIndex = e2.preIndex;
	strenth = e2.strenth;
}

void PreSpkEvent::operator=(const PreSpkEvent & e2){
	__super::operator=(e2);
	preIndex = e2.preIndex;
}

void CheckEvent::operator=(const CheckEvent & e2){
	__super::operator=(e2);
	index = e2.index;
}
