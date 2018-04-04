#ifndef TIGR_UTILITIES_HPP
#define TIGR_UTILITIES_HPP


#include "globals.hpp"

namespace utilities {
	void PrintResults(uint *results, uint n);
	void PrintResults(float *results, uint n);
	void SaveResults(string filepath, uint *results, uint n);
	void SaveResults(string filepath, float *results, uint n);
}

#endif	//	TIGR_UTILITIES_HPP
