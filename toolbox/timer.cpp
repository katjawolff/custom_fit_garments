#include "timer.h"
#include <iostream>


Timer::Timer(const std::string& str)
: start(clock::now()), name(str), sum(clock::duration::zero())
{
}

Timer::clock::duration
Timer::elapsed() const
{
    if(isDisabled) return Timer::clock::duration(0);
    
    if(isPaused) return sum;
    else return clock::now() - start + sum;
}

void
Timer::disable()
{
    isDisabled = true;
}


void
Timer::enable()
{
    isDisabled = false;
}


void
Timer::pause()
{
    if(isDisabled) return;
    
    if(!isPaused)
    {
        isPaused = true;
        sum += clock::now() - start;
    }
}

void
Timer::resume()
{
    if(isDisabled) return;
    
    if(isPaused)
    {
        isPaused = false;
        start = clock::now();
    }
}

void
Timer::reset()
{
    if(isDisabled) return;
    
    sum = Timer::clock::duration::zero();
    start = clock::now();
}

int
Timer::seconds() const
{
    return static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(elapsed()).count());
}

int
Timer::milliseconds() const
{
    if(isDisabled) return 0;
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed()).count());
}

long long
Timer::microseconds() const
{
    if(isDisabled) return 0;
    return static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed()).count());
}

int
Timer::hours() const
{
    return static_cast<int>(std::chrono::duration_cast<std::chrono::hours>(elapsed()).count());
}

int
Timer::minutes() const
{
    return static_cast<int>(std::chrono::duration_cast<std::chrono::minutes>(elapsed()).count());
}

void
Timer::printTime(const std::string& str)
{
#ifdef PRINT_TIME
    
    if(isDisabled) return;
    
    using namespace std::chrono;

    const auto diff = elapsed();

    int us = static_cast<int>(duration_cast<std::chrono::microseconds>(diff).count());
    int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
    int s = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(diff).count());
    int m = static_cast<int>(std::chrono::duration_cast<std::chrono::minutes>(diff).count());
    int h = static_cast<int>(std::chrono::duration_cast<std::chrono::hours>(diff).count());

    us -= 1000 * ms;
    ms -= 1000 * s;
    s -= 60 * m;
    m -= 60 * h;

    std::cout << name << " " << str << ": ";
    if(h) std::cout << h << "h ";
    if(m || h) std::cout << m << "m ";
    if(s || m || h) std::cout << s << "s ";
    if(s || m || h || ms) std::cout << ms << "ms ";

    std::cout << us << "μs" << std::endl << std::flush;
    
    reset();
    
#endif
}
