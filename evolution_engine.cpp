/*
Score-based natural selection engine

Author: xakepp35@gmail.com
Date: Feb 2018
License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#include "evolution_engine.h"

//#include <random>		// std::uniform_real_distribution, random generation
#include <numeric>		// std::iota, not to invent bicycles
#include <algorithm>    // std::sort, not to write code yourself
#include <iostream>

#include <ppl.h>

using namespace std;

namespace evolution {
	engine::engine(bool multiThreaded, size_t orderPower, size_t generationStepMax, uint64_t rndSeed):
		_multiThreaded(multiThreaded),
		_orderPower( orderPower ),
		_generationStepMax(generationStepMax),
		_generationStep(0),
		_rGen( rndSeed )
	{
		_vScore.resize(((size_t)1) << orderPower, 0);
		_vScoreIndex.resize(_vScore.size());
		_nGeneration = 0;
	}

	void engine::initial_agent_spawn() {
		_vAgent.reserve(_vScore.size());
		for (size_t i = 0; i < _vScore.size(); i++)
			_vAgent.emplace_back(randomized_agent(_rGen));
	}

	
	void engine::step() {
		if(_multiThreaded)
			concurrency::parallel_for(size_t(0), num_agents(), [this](size_t i) {
				_vScore[i] += _vAgent[i]->step(this);
			});
		else
			for (size_t i = 0; i < num_agents(); i++)
				_vScore[i] += _vAgent[i]->step(this);

		_generationStep++;
		if (_generationStepMax != 0)
			if (_generationStep >= _generationStepMax)
				new_generation();
	}

	void engine::new_generation() {
		reevaluate_scores_indices();
		scalar overallScore = 0;
		for (auto& i : _vScore)
			overallScore += i;
		std::cout << "gen#" << _nGeneration << " score: top=" << _vScore[_vScoreIndex[0]] << "\tavg=" << overallScore / _vScore.size() << "\n";

		_vAgent = run_selection_step();
		_nGeneration++;
		

		reset_scores();
		_generationStep = 0;
	}


	size_t engine::num_agents() const {
		return _vAgent.size();
	}

	void engine::reset_scores() {
		for (auto& fScore : _vScore)
			fScore = 0;
	}

	template <typename T> vector<size_t> sort_indexes(const vector<T> &v) {
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);
		sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
		return idx;
	}

	void engine::reevaluate_scores_indices() {
		iota(_vScoreIndex.begin(), _vScoreIndex.end(), 0);
		sort(_vScoreIndex.begin(), _vScoreIndex.end(), [this](size_t i0, size_t i1) {return _vScore[i0] > _vScore[i1]; });
	}
	

	vector< agent::ptr > engine::run_selection_step() {
		
		vector< agent::ptr > newAgent;
		newAgent.reserve(_vAgent.size());
		for (size_t i = 0; i < _orderPower; i++) { // we would take k winners
			auto numChildren = ((size_t)1) << (_orderPower - i - 1);
			for (size_t j = 0; j < numChildren; j++) { // and each winner would have log2()-ically decreasing number of children, dependent on its place in score chart
				newAgent.emplace_back(_vAgent[_vScoreIndex[i]]->crossover(randomized_agent(_rGen), ((scalar)j) / (numChildren)) ); // j/numChildren is mutation percent, eg original agent 0% mutated, 1% mutated, ...
				// uncomment to see how this calculus works in action:
				// std::cout << "i=" << i << "\tj=" << j << "\n";
			}
		}
		// here we have 1 unfilled agent, i recommend doing this, as this could speedup evolution by taking best: 2/3 from the leader and 1/3 from second place genes
		newAgent.emplace_back(_vAgent[_vScoreIndex[0]]->crossover(_vAgent[_vScoreIndex[1]], ((scalar)1) / 3));
		return newAgent;
	}


}