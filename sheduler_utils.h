#pragma once

#include <cstdint>

// singletone, OS-specific threading and sheduling functions
class sheduler {
public:

	typedef uint64_t stamp;

	static stamp now();	// obtains current timestamp
	static stamp freq(); // frequency = now much stamp ticks per one second (stamp change rate)

	template< typename T>
	inline static T map_time(const stamp& dS) {
		return static_cast<T>(dS) / freq();
	}
	
	// puts thread to sleep
	static void yield();


};
