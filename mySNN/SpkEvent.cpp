#include "SpkEvent.h"
#include <limits>

bool operator<(const PostSpkEvent & e1, const PostSpkEvent & e2){
	if (e1.time == e2.time) {
		if (e1.preIndex == e2.preIndex) {
			if (e1.layer == e2.layer) {
				return e1.postIndex < e2.postIndex;
			}
			return e1.layer < e2.layer;
		}
		return e1.preIndex < e2.preIndex; // case e1.preIndex == UINT_MAX need to be solve after others
	}
	return e1.time < e2.time;
}

bool operator>(const PostSpkEvent & e1, const PostSpkEvent & e2) {
	if (e1.time == e2.time) {
		if (e1.preIndex == e2.preIndex) {
			if (e1.layer == e2.layer) {
				return e1.postIndex > e2.postIndex;
			}
			return e1.layer > e2.layer;
		}
		return e1.preIndex > e2.preIndex; // case e1.preIndex == UINT_MAX need to be solve after others
	}

	return e1.time > e2.time;
}
