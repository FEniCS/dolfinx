// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2007-05-14

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include "Logger.h"

using namespace dolfin;

typedef std::map<std::string, std::pair<dolfin::uint, real> >::iterator map_iterator;

//-----------------------------------------------------------------------------
Logger::Logger()
  : destination(terminal), debug_level(0), indentation_level(0), logstream(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Logger::message(std::string msg, int debug_level)
{
  if (debug_level > this->debug_level)
    return;

  write(debug_level, msg);
}
//-----------------------------------------------------------------------------
void Logger::warning(std::string msg)
{
  std::string s = std::string("*** Warning: ") + msg;
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::error(std::string msg)
{
  std::string s = std::string("*** Error: ") + msg;
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::begin(std::string msg, int debug_level)
{
  // Write a message
  message(msg, debug_level);
  indentation_level++;
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  indentation_level--;
}
//-----------------------------------------------------------------------------
void Logger::progress(std::string title, real p)
{
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = static_cast<int>(p*static_cast<real>(N));
  
  // Print the title
  std::string s = "| " + title;
  for (uint i = 0; i < (N - title.size() - 1); i++)
    s += " ";
  s += "|";
  write(0, s);
  
  // Print the progress bar
  s = "|";
  for (int i = 0; i < n; i++)
    s += "=";
  if (n > 0 && n < N)
  {
    s += "|";
    n++;
  }
  for (int i = n; i < N; i++)
    s += "-";
  s += "| ";
  std::stringstream line;
  line << std::setiosflags(std::ios::fixed);
  line << std::setprecision(1);
  line << 100.0*p;
  s += line.str() + "%";
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::setOutputDestination(std::string destination)
{
  // Choose output destination
  if (destination == "terminal")
    this->destination = terminal;
  else if (destination == "silent")
    this->destination = silent;
  else if (destination == "stream"){
    warning("Please provide the actual stream. Using terminal instead.");
    this->destination = terminal;
  }
  else
  {
    this->destination = terminal;
    message("Unknown output destination, using plain text.");
  }
}
//-----------------------------------------------------------------------------
void Logger::setOutputDestination(std::ostream& ostream)
{
   logstream = &ostream;
   this->destination = stream;
}
//-----------------------------------------------------------------------------
void Logger::setDebugLevel(int debug_level)
{
  this->debug_level = debug_level;
}
//-----------------------------------------------------------------------------
void Logger::timing(std::string task, real elapsed_time)
{
  // Print a message
  std::stringstream line;
  line << "Elapsed time: " << elapsed_time << " (" << task << ")";
  message(line.str());

  // Store values for summary
  map_iterator it = timings.find(task);
  if (it == timings.end())
  {
    std::pair<uint, real> timing(1, elapsed_time);
    timings[task] = timing;
  }
  else
  {
    it->second.first += 1;
    it->second.second += elapsed_time;
  }
}
//-----------------------------------------------------------------------------
void Logger::summary()
{
  if (timings.size() == 0)
  {
    message("Summary: no timings to report.");
    return;
  }

  message("Summary of timings:");
  for (map_iterator it = timings.begin(); it != timings.end(); ++it)
  {
    const std::string task  = it->first;
    const uint num_timings  = it->second.first;
    const real total_time   = it->second.second;
    const real average_time = total_time / static_cast<real>(num_timings);

    std::stringstream line;
    line << "  " << task << ": " << total_time << " " << average_time << " " << num_timings;
    message(line.str());
  }
}
//-----------------------------------------------------------------------------
void Logger::__debug(std::string msg)
{
  std::string s = std::string("Debug: ") + msg;
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::__assert(std::string msg)
{
  std::string s = std::string("*** Assertion ") + msg;
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::write(int debug_level, std::string msg)
{
  // Check debug level
  if (debug_level > this->debug_level)
    return;

  // Add indentation
  for (int i = 0; i < indentation_level; i++)
    msg = "  " + msg;
  
  // Choose destination
  switch (destination)
  {
  case terminal:
    std::cout << msg << std::endl;
    break;
  case stream:
    if (logstream == NULL)
      error("No stream attached, cannot write to stream");
    *logstream << msg << std::endl;
    break;
  default:
    // Do nothing if destination == silent
    do {} while (false);
  }
}
//----------------------------------------------------------------------------
