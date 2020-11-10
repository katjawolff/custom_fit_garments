#pragma once

#include <chrono>
#include <string>

struct Timer
{
    typedef std::chrono::system_clock clock;
    typedef std::chrono::time_point<clock> timepoint;

    timepoint start;
    std::string name;
    clock::duration sum;
    bool isPaused = false;
    bool isDisabled = false;
    
    explicit Timer(const std::string& str = "Timer");

    void disable();
    void enable();
       
    void pause();
    void resume();
    clock::duration elapsed() const;

    void reset();
    int seconds() const;
    int milliseconds() const;
    long long microseconds() const;
    int minutes() const;
    int hours() const;
    void printTime(const std::string& str = "") ;
};


