#include "sheduler_utils.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>


sheduler::stamp sheduler::now() {
	stamp hpc = 0;
	QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&hpc));
	return hpc;
}


sheduler::stamp sheduler::freq() {
	stamp hpf = 0;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&hpf));
	return hpf;
}


void sheduler::yield() {
	SleepEx(1, false);
}
