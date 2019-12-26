
#include "timer.hpp"


void Timer::Start()
{
	gettimeofday( &StartingTime, NULL );
}


float Timer::Finish()
{
	timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	float d = ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;
	return d;
}
