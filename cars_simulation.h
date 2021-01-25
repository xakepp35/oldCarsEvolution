/*
Demonstration of perceptrone learning

Author: xakepp35@gmail.com
Date: Feb 2018
License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#pragma once


#include "neural_net.h"
#include "physics_engine.h"
#include "evolution_engine.h"


namespace simulation {
		
// glue class; represents single car and glues several diferent paradigms together
class car:
	public evolution::agent
{
public:

	car(neural::net&& newNet);
	virtual ~car() override;
	virtual evolution::scalar step(evolution::engine* enviromentEngine) override;
	virtual ptr crossover(const ptr& dstAgent, const evolution::scalar& dstAlpha) override;

	physics::vehicle	_mPhysics; // physics model
	float				_pAtan;

//protected:

	neural::net			_mNeural; // neural model
	
	

};

class engine :
	public evolution::engine
{
public:

	engine(bool multiThreaded, size_t orderPower, size_t generationStepMax, float fixedDT);

	virtual evolution::agent::ptr randomized_agent(xoroshiro128plus& rGen) override;
	float _weightDelta;
	
	physics::engine	_mPhysics;

	// simple rectangular track with thin tubes
	void generate_rectangle_track();

	// circular track procedural generation
	void generate_track(xoroshiro128plus& rGen);
	
	std::vector< scalar > _sensorAngles;
};


}