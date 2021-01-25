#include "neural_net.h"
#include <random>		// std::uniform_real_distribution, random generation

#include "fast_math.h"


namespace neural {

////////////////////////////////////////////////////////////////
//	cell
////////////////////////////////////////////////////////////////

	cell::cell() {}
		
	cell cell::randomized( size_t nInputs, const scalar& wBias, xoroshiro128plus& srcGenerator, const scalar& wMin, const scalar& wMax) {
		cell outResult;
		outResult._wBias = wBias;
		std::uniform_real_distribution<scalar> uDweights(wMin, wMax);
		outResult._vWeight.resize(nInputs / 4);
		for (size_t i = 0; i < nInputs; ++i)
			reinterpret_cast<scalar*>(outResult._vWeight.data())[i]=uDweights(srcGenerator);
		return outResult;
	}

	static inline scalar lerp(const scalar& x0, const scalar& x1, const scalar& a) {
		//return x0 * (((scalar)1) - a) + x1 * a;
		return x0 + x1 * a;
	}

	cell cell::crossover(const cell& targetcell, const scalar& targetAlpha) const {
		cell outResult;
		outResult._wBias = lerp(_wBias, targetcell._wBias, targetAlpha);
		outResult._vWeight.resize(num_inputs() / 4);
		for (size_t i = 0; i < num_inputs(); ++i)
			reinterpret_cast<scalar*>(outResult._vWeight.data())[i] = lerp(reinterpret_cast<const scalar*>(&_vWeight[0])[i], reinterpret_cast<const scalar*>(&targetcell._vWeight[0])[i], targetAlpha);
		return outResult;
	}

	/*
	cell cell::crossover(const cell& targetcell, const scalar& targetAlpha) const {
		cell outResult;
		outResult._wBias = lerp(_wBias, targetcell._wBias, targetAlpha);
		outResult._vWeight.reserve(num_inputs());
		for (size_t i = 0; i < num_inputs(); ++i)
			outResult._vWeight.emplace_back(lerp(reinterpret_cast<const scalar*>(&_vWeight[0])[i], reinterpret_cast<const scalar*>(&targetcell._vWeight[0])[i], targetAlpha));
		return outResult;
	}
	*/

	//__declspec(noinline)
		scalar cell::calc(const __m128* inputData) const {
			return vdot4(inputData, _vWeight.data()); 
	}
	/*
	__declspec(noinline)
	scalar cell::calc(const scalar* inputData) const {
		// apply bias if any
		auto outResult = _wBias; // could really help solve tasks where threshold is required to fine tune. try to zero it and it will slow down/stop evolving

		// calculate dot product
		for (size_t i = 0; i < _vWeight.size(); i++)
		outResult += _vWeight[i] * inputData[i];

		// apply squashing function

		//return posexp_sigmf(outResult); // REALLY FUCK THIS!!!
		//return tanh(outResult); // TANH rocks in vehicle/robotics control!
		//return faster_tanh(outResult);
		//return linv_sigmf(outResult); // works so-so
		//return sqinv_sigmf(outResult);
		return fastest_tanh(outResult);
	}*/


	size_t cell::num_inputs() const {
		return _vWeight.size() * 4;
	}

////////////////////////////////////////////////////////////////
//	barrel_mem
////////////////////////////////////////////////////////////////

	barrel_mem::barrel_mem(size_t nLength, size_t nTapes) {
		_tapeIndex = 0;
		_vMemory.reserve(nTapes);
		for (size_t i = 0; i < nTapes; i++)
			_vMemory.emplace_back(nLength/4);
	}

	
						   
	const __m128* barrel_mem::data() const {
		return _vMemory[_tapeIndex].data();
	}

	__m128* barrel_mem::data() {
		return _vMemory[_tapeIndex].data();
	}

	size_t barrel_mem::size() const {
		return _vMemory[_tapeIndex].size();
	}

	size_t barrel_mem::num_tapes() const
	{
		return _vMemory.size();
	}

	__m128* barrel_mem::swap_tapes() {
		auto inputTape = data();
		_tapeIndex += 1;
		_tapeIndex %= _vMemory.size();
		return inputTape;
	}

////////////////////////////////////////////////////////////////
//	net
////////////////////////////////////////////////////////////////

	net::net(size_t ncells, size_t nInputs, size_t nTapes):
		barrel_mem(ncells+nInputs, nTapes),
		_vcell(ncells)
	{}

	const scalar* net::input_data() const {
		return reinterpret_cast<const scalar*>(data()) + _vcell.size();
	}

	scalar* net::input_data() {
		return reinterpret_cast<scalar*>( data()) + _vcell.size();
	}

	size_t net::input_size() const {
		return size()*4 - _vcell.size();
	}

	net net::randomized(size_t nCells, size_t nInputs, size_t nTapes, xoroshiro128plus& srcGenerator, const scalar& wMin, const scalar& wMax, const scalar& wBiasMin, const scalar& wBiasMax) {
		net outResult(nCells, nInputs, nTapes);
		std::uniform_real_distribution<scalar> uDbias(wBiasMin, wBiasMax);
		for( size_t i = 0; i < nCells; i++)
			outResult._vcell[i] = std::move( cell::randomized(nCells+nInputs, uDbias(srcGenerator), srcGenerator, wMin, wMax) );
		return outResult;
	}

	net net::crossover(const net& dstNet, const scalar& targetAlpha) const {
		auto ncells = _vcell.size();
		net outResult(ncells, input_size(), num_tapes());
		for (size_t i = 0; i < ncells; i++)
			outResult._vcell[i] = _vcell[i].crossover(dstNet._vcell[i], targetAlpha);
		return outResult;
	}

	__m128* net::step() {
		auto inputTape = swap_tapes();
		auto scalarArray = reinterpret_cast<scalar*>(data());
		for (size_t i = 0; i < _vcell.size(); i++)
			scalarArray[i] = _vcell[i].calc(inputTape);
		for (size_t i = 0; i < _vcell.size(); i++)
			scalarArray[i] = fastest_tanh(scalarArray[i]);
		return inputTape;
	}

}