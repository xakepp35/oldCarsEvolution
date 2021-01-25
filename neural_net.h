/*
Neural network ulitities

Author: xakepp35@gmail.com
Date: Feb 2018
License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#pragma once

#include <cstdint>		// all standart integer types
#include <vector>       // std::vector, storage container and memory management

#include "xoroshiro128plus.h"
#include "fast_math.h"


namespace neural {

	// scalar data type we prefer to use
	typedef float scalar;

	// everything you need to manage and calculate single cell
	class cell {
	public:

		// constructs empty cell
		cell();

		// generates random cell
		static cell randomized(size_t nInputs, const scalar& wBias, xoroshiro128plus& srcGenerator, const scalar& wMin, const scalar& wMax);

		// crossover function, linear interpolation = this->_weightVec * (1-alpha) + destAgent._weightVec * alpha; it is assumed that target cell has same number of weights.
		cell crossover(const cell& targetcell, const scalar& targetAlpha) const;

		// given input, calculates cell output
		scalar calc(const __m128* inputData) const;

		// returns number of weights
		size_t num_inputs() const;

	protected:

		scalar					_wBias;
		std::vector< __m128 >	_vWeight;

	};

	
	// swappable memory tapes for I/O. very powerful shit, like a gatling gun.
	// memory tape organisation is quite tricky here. but it is a simple and awesome idea
	// basically at lower level, it has 2 memory regions, at every step one is considered read only, second is write only. 
	// and they do swap their roles every step. this number can be extended further, to record network operation history for N steps and making analysis easy: full input and output for each step.
	// this can afford network to read its output from previous step each time, in a parallel for each cell, while writing next output without messing up.
	class barrel_mem {
	public:

		barrel_mem(size_t nLength, size_t nTapes = 2);

		//after the step you may read output values
		const __m128* data() const; // returns pointer to network output data
		__m128* data(); // returns pointer to network output data
		size_t size() const; // returns number of elements
		size_t num_tapes() const; // returns number of tapes. stick with 2, until otherwise is really required

	protected:

		// to be used internally befure actual step, returns new output tape
		__m128* swap_tapes();
	public:
		std::vector< std::vector< __m128 > >	_vMemory;
		size_t									_tapeIndex;

	};

	// represents fully connected neural network, where each cell is connected to all inputs and all previous step's results' outputs
	// it can be thought as a single layer with a 1-tick delayed feedback.
	class net:
		public barrel_mem
	{
	public:

		net(size_t ncells, size_t nInputs, size_t nTapes = 2);


		// constructs fully random network
		static net randomized(size_t nCells, size_t nInputs, size_t nTapes, xoroshiro128plus& srcGenerator, const scalar& wMin, const scalar& wMax, const scalar& wBiasMin, const scalar& wBiasMax);

		// spawns a network by LERPing current to dest
		net crossover(const net& dstNet, const scalar& targetAlpha) const;

		// fill input data before step!
		const scalar* input_data() const; // returns pointer to network input data
		scalar* input_data(); // returns pointer to network input data
		size_t input_size() const; // returns count of inputs

		// then call actual calculation step, that simple. after step - read net::data() to get outputs
		// step() returns input tape, as there will be no way of getting it after. we dont need it after step, so dont use this return value!
		__m128* step();


	//protected:

		std::vector<cell>		_vcell;
		
	};



}