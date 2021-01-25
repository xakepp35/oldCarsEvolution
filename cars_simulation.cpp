/*
	Demonstration of perceptrone learning

	Author: xakepp35@gmail.com
	Date: Feb 2018
	License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#include "cars_simulation.h"

#include <random>		// std::uniform_real_distribution, random generation
#include <iostream>     // std::cout, printing results


using namespace std;


namespace simulation {

	car::car(neural::net&& newNet):
		_mNeural(newNet),
		_mPhysics()
	{
		// for simplicity default parameters goes here
		_mPhysics.p = point( 0.0f, 0.8f );
		_mPhysics.p0 = _mPhysics.p;
		_mPhysics.ang = 0;
		_mPhysics.ang0 = 0;
		_mPhysics.rSqr = 1.0f / 64;
		_mPhysics.rSqr *= _mPhysics.rSqr;
		_mPhysics._isActive = true;
		_pAtan = atan2(_mPhysics.p[1], _mPhysics.p[0]);
	}

	car::~car() {
	}

	evolution::scalar car::step(evolution::engine * enviromentEngine) {
		if (!_mPhysics._isActive)
			return 0;
		auto& myEngine = *static_cast<engine*>(enviromentEngine);

		//auto cAtan = atan2(_mPhysics.p[1], _mPhysics.p[0]);
		auto cAtan = fastest_atan2(_mPhysics.p[1], _mPhysics.p[0]);
		
		auto dAtan = cAtan - _pAtan;
		_pAtan = cAtan;
		while (dAtan >= pi)
			dAtan -= 2 * pi;
		while (dAtan < -pi)
			dAtan += 2 * pi;

		// collision = penalty
		//if (!_mPhysics._isActive) {
		//	return (physics::pi - cAtan) * myEngine._mPhysics.get_dt() / 8;
			//return -1.0f / 512;
			//std::uniform_int_distribution< size_t > uDi(0, _mNeural._vcell.size());			
			//_mNeural._vcell[ uDi(myEngine._rGen)].crossover(neural::cell::randomized(_mNeural._vcell.size() + _mNeural.input_size(), 0, myEngine._rGen, -1, 1), 1.0f / 8);
		//}
		//return 0;// physics::pi - cAtan; // crashed car, reward each step


		//float totalScore = 0;
		
		// copy visual field input data
		myEngine._mPhysics.calculate_sensor_proximity_field(_mPhysics, _mNeural.input_data(), myEngine._sensorAngles);
		_mNeural.step();
		// tanh() makes signed control possible => lower neuron count
		_mPhysics._vehicleControls[physics::vehicle::SteeringControl] = ((scalar*)_mNeural.data())[0]; // -_mNeural.data()[1];
		_mPhysics._vehicleControls[physics::vehicle::MotionControl] = ((scalar*)_mNeural.data())[1]; // -_mNeural.data()[3] / 2;
		_mPhysics.step(myEngine._mPhysics);

		//auto spdsqr = _mPhysics.get_speed();
		//spdsqr *= spdsqr;
		/*
		auto rotsqr = _mPhysics._vehicleControls[physics::vehicle::SteeringControl] / 20;
		rotsqr *= rotsqr;
		*/

		return 0
			+ 1.0f / 16384 // bonus for staying alive
			// +_mPhysics._vehicleControls[physics::vehicle::MotionControl] // bonus for pressing gas moving forward
			// +spdsqr // square of velocity; faster moving cars = greater bonus; forces car to cycle back and forth an a straight line, and not to go in hard turns and tiny passages
			// -rotsqr // punishment for excess fotation. bad, forces in not to rotate swiftly in a hard place
			// +pi - cAtan // bonus for angular position on the track. bad, because we have to integrate atan deltas, but not giving absolute atan2() value..
			-dAtan // BEST!!! arctangent delta between this frame and prev frame; kind of angular velocity through the track; when a full track is passed that is delta = 6.28; forces increase circular track path passed
			;
	}

	car::ptr car::crossover(const ptr & dstAgent, const evolution::scalar & dstAlpha)
	{
		auto dstCar = static_pointer_cast<car>(dstAgent);
		auto newCar = make_shared< car >(std::move(_mNeural.crossover(dstCar->_mNeural, dstAlpha)));
		// default newCar->_mPhysics
		return newCar;
	}

	engine::engine(bool multiThreaded, size_t orderPower, size_t generationStepMax, float fixedDT):
		evolution::engine(multiThreaded, orderPower, generationStepMax, xoroshiro128plus::defaultSeed ),
		_mPhysics(fixedDT)
	{
		_weightDelta = 1.0f;
		_sensorAngles = _mPhysics.build_sensor_angles(2);
	}

	evolution::agent::ptr engine::randomized_agent(xoroshiro128plus& rGen) {
		
		// worked after 100-150 gens, with datan only:
		//auto newCar = make_shared< car >(neural::net::randomized(16, 2, 2, rGen, -1.0/4, 1.0/4, -1.0/2, 1.0/2));
		auto newCar = make_shared< car >(neural::net::randomized(2, 2, 2, rGen, -_weightDelta, _weightDelta, 0, 0));// -1.0 / 128, 1.0 / 128));

		//auto newCar = make_shared< car >(neural::net::randomized(12, 4, 2, rGen, -1, 1, -1, 1));
		//auto newCar = make_shared< car >(neural::net::randomized(4, 3, 2, rGen, -4, 4, -4, 4));
		//auto newCar = make_shared< car >(neural::net::randomized(16, 2, 2, rGen, -16, 16, -16, 16));
		//auto newCar = make_shared< car >(neural::net::randomized(16, 2, 2, rGen, -16, 16, -16, 16));
		return newCar;

	}

	void engine::generate_rectangle_track() {
		auto& vWalls = _mPhysics._vWalls;
		vWalls.emplace_back(physics::segment_from_points( point( 0.9f,0.9f ),		point( 0.9f,-0.9f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( 0.9f,-0.9f ),		point( -0.9f,-0.9f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( -0.9f,-0.9f ),		point( -0.9f,0.9f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( -0.9f,0.9f ),		point( 0.9f,0.9f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( 0.7f,0.7f ),		point( 0.7f,-0.8f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( 0.7f,-0.8f ),		point( -0.85f,-0.8f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( -0.85f,-0.8f ),	point( -0.85f,0.7f ) ));
		vWalls.emplace_back(physics::segment_from_points( point( -0.85f,0.7f ),		point( 0.7f,0.7f ) ));
	}

	// vector<physics::segment> vWalls;
	void engine::generate_track(xoroshiro128plus& rGen) {
		auto& vWalls = _mPhysics._vWalls;
		float px = 1;
		float py = 0;
		std::uniform_real_distribution< float > uDi(0.15f, 0.25f);
		float prInner = 0.8f - 0.2f;
		float prOuter = 0.8f + 0.2f;
		size_t numTrackSegments = 29;


		for (size_t i = 1; i <= numTrackSegments; i++) {
			auto angl = (pi * (2 * i)) / numTrackSegments;
			auto x = cos(angl);
			auto y = sin(angl);

			float rInner = 0.8f - uDi(rGen);
			float rOuter = 0.8f + uDi(rGen);
			if (i == numTrackSegments) {
				rInner = 0.6f;
				rOuter = 1.0f;
			}

			vWalls.emplace_back(physics::segment_from_points( point(px*prInner, py*prInner ), point( x*rInner, y*rInner ) ));
			vWalls.emplace_back(physics::segment_from_points( point(px*prOuter, py*prOuter ), point( x*rOuter, y*rOuter ) ));
			if (static_cast<int64_t>(rGen()) > 0) {
				if (static_cast<int64_t>(rGen()) < 0)
					vWalls.emplace_back(physics::segment_from_points( point(x*0.85f, y*0.85f ), point( x*rInner, y*rInner ) ));
				else
					vWalls.emplace_back(physics::segment_from_points( point(x*0.75f, y*0.75f ), point( x*rOuter, y*rOuter ) ));
			}

			px = x; py = y;
			prInner = rInner; prOuter = rOuter;
		}
	}


}

