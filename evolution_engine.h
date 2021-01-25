/*
Score-based natural selection engine

Author: xakepp35@gmail.com
Date: Feb 2018
License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#pragma once

#include <cstdint>		// all standart integer types
#include <vector>       // std::vector, storage container and memory management
#include <memory>
#include "xoroshiro128plus.h"


namespace evolution {

	// scalar type we want to use here
	typedef float scalar;

	// engine prototype
	class engine;
		
	// agent interface. there could be alot of things inside simulation, like physics, neural networking, so on. 
	// so we will stick with abstract interfaces to make system agile and suitable to any task
	class agent {
	public:

		// automatic memory management is convenient
		typedef std::shared_ptr< agent > ptr;

		// virtual abstract destructor is required here to get rid of potential memory leaks
		virtual ~agent() {};

		// performs simulation step for an agent, must return its score per step
		virtual scalar step(engine* enviromentEngine) = 0;

		// we have to have some crossover function, in order to define evolution algorithm it terms of it
		virtual ptr crossover(const ptr& dstAgent, const scalar& dstAlpha) = 0;

	};

	class engine {
	public:

		// creates 2 to the power orderPower randomized agents. you should overload randomized_agent() for that
		engine(bool multiThreaded, size_t orderPower, size_t generationStepMax = 0, uint64_t rndSeed = xoroshiro128plus::defaultSeed);
		void initial_agent_spawn();

		virtual agent::ptr randomized_agent(xoroshiro128plus& rGen) = 0;

		// runs one step for all agents
		void step();

		// create new generation, that is greater/faster/stronger
		void new_generation(); // called internally when number of steps exceeds max

		size_t num_agents() const;

		std::vector< agent::ptr > _vAgent;

	protected:

		void reset_scores();

		void reevaluate_scores_indices();
		
		std::vector< agent::ptr > run_selection_step();


	//protected:
	public:
		
		std::vector< scalar > _vScore;
		std::vector< size_t > _vScoreIndex;

		xoroshiro128plus _rGen;
		size_t	_orderPower; // we will have 2^_orderPower agents total
		size_t	_nGeneration;

		size_t _generationStepMax;
		size_t _generationStep;
		bool _multiThreaded;
	};



}