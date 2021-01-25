/*
xoroshiro128+ random generator c++ wrapper

author webpage: http://xoroshiro.di.unimi.it/
source code: http://vigna.di.unimi.it/xorshift/xoroshiro128plus.c

Author: xakepp35@gmail.com
Date: Feb 2018
License: FreeBSD/ISC for non commercial (personal/educational) use
*/

#pragma once

#include <cstdint>

// good quality and fast random generator - a key to success :D
class xoroshiro128plus {
public:

	static const uint64_t defaultSeed = 1234567898987654321ULL;

	// initial seed must be nonzero, and well randomized
	xoroshiro128plus(uint64_t initialSeed = defaultSeed);

	static inline uint64_t min() { return 0; }
	static inline uint64_t max() { return 0xFFFFFFFFFFFFFFFFULL; }

	typedef uint64_t result_type;

	// xoroshiro128+ generator
	uint64_t operator()();

	// 2^64 calls to the operator(), can generate 2^64 non-overlapping sequences for parallel computation
	void jump();

protected:

	uint64_t s[2];

};
