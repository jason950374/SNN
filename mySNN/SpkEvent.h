#pragma once

enum EventIDList { EventID , PostSpkEventID, PreSpkEventID, CheckEventID};

class Event {
public:
	mutable EventIDList ID = EventID;
	double time;
	unsigned int layer; //belong to post-neuron
	bool operator <(const Event& e2) const;
	bool operator >(const Event& e2) const;
	virtual void operator =(const Event& e2); // Pass by value. If want the same instance, use pointer instead.
	virtual ~Event() {};
};

class PostSpkEvent : public Event {
public:
	PostSpkEvent() { ID = PostSpkEventID; }
	//location
	unsigned int postIndex;
	unsigned int preIndex; //for backward
	double strenth;
	void operator =(const PostSpkEvent& e2); // Pass by value. If want the same instance, use pointer instead.
};

//temparary event
//turn to PostSpkEvent after Synapse::preSynEvent
class PreSpkEvent : public Event {
public:
	PreSpkEvent() { ID = PreSpkEventID; }
	unsigned int preIndex;
	void operator =(const PreSpkEvent& e2); // Pass by value. If want the same instance, use pointer instead.
};

class CheckEvent : public Event {
public:
	CheckEvent() { ID = CheckEventID; }
	unsigned int index;
	void operator =(const CheckEvent& e2); // Pass by value. If want the same instance, use pointer instead.
};