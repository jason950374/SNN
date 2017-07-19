#include "SpkEvent.h"

bool operator<(const PostSpkEvent & e1, const PostSpkEvent & e2){
	return e1.time < e2.time;
}

bool operator>(const PostSpkEvent & e1, const PostSpkEvent & e2) {
	return e1.time > e2.time;
}
