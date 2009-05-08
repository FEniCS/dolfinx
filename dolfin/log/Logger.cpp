// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2009-05-08

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include <dolfin/main/MPI.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include "Table.h"
#include "Logger.h"

using namespace dolfin;

typedef std::map<std::string, std::pair<dolfin::uint, double> >::iterator map_iterator;
typedef std::map<std::string, std::pair<dolfin::uint, double> >::const_iterator const_map_iterator;

//-----------------------------------------------------------------------------
Logger::Logger()
  : destination(terminal), debug_level(0), indentation_level(0), logstream(&std::cout),
    process_number(-1)
{
  if (MPI::num_processes() > 1)
    process_number = MPI::process_number();
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Logger::info(std::string msg, int debug_level) const
{
  if (debug_level > this->debug_level)
    return;

  write(debug_level, msg);
}
//-----------------------------------------------------------------------------
void Logger::info_underline(std::string msg, int debug_level) const
{
  if (msg.size() == 0)
    info(msg, debug_level);

  std::stringstream s;
  s << msg;
  s << "\n";
  for (int i = 0; i < indentation_level; i++)
    s << "  ";
  for (uint i = 0; i < msg.size(); i++)
    s << "-";

  info(s.str(), debug_level);
}
//-----------------------------------------------------------------------------
void Logger::warning(std::string msg) const
{
  std::string s = std::string("*** Warning: ") + msg;
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::error(std::string msg) const
{
  std::string s = std::string("*** Error: ") + msg;
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::begin(std::string msg, int debug_level)
{
  // Write a message
  info(msg, debug_level);
  indentation_level++;
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  indentation_level--;
}
//-----------------------------------------------------------------------------
void Logger::progress(std::string title, double p) const
{
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = static_cast<int>(p*static_cast<double>(N));

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
void Logger::set_output_destination(std::string destination)
{
  // Choose output destination
  if (destination == "terminal")
  {
    this->destination = terminal;
    logstream = &std::cout;
  }
  else if (destination == "silent")
    this->destination = silent;
  else if (destination == "stream")
  {
    warning("Please provide the actual stream. Using terminal instead.");
    this->destination = terminal;
    logstream = &std::cout;
  }
  else
  {
    this->destination = terminal;
    logstream = &std::cout;
    info("Unknown output destination, using plain text.");
  }
}
//-----------------------------------------------------------------------------
void Logger::set_output_destination(std::ostream& ostream)
{
   logstream = &ostream;
   this->destination = stream;
}
//-----------------------------------------------------------------------------
void Logger::set_debug_level(int debug_level)
{
  this->debug_level = debug_level;
}
//-----------------------------------------------------------------------------
void Logger::register_timing(std::string task, double elapsed_time)
{
  // Remove small or negative numbers
  if (elapsed_time < DOLFIN_EPS)
    elapsed_time = 0.0;

  // Print a message
  std::stringstream line;
  line << "Elapsed time: " << elapsed_time << " (" << task << ")";
  info(line.str(), 1);

  // Store values for summary
  map_iterator it = timings.find(task);
  if (it == timings.end())
  {
    std::pair<uint, double> timing(1, elapsed_time);
    timings[task] = timing;
  }
  else
  {
    it->second.first += 1;
    it->second.second += elapsed_time;
  }
}
//-----------------------------------------------------------------------------
void Logger::summary(bool reset)
{
  if (timings.size() == 0)
  {
    info("Summary: no timings to report.");
    return;
  }

  info("");
  Table table("Summary of timings");
  for (const_map_iterator it = timings.begin(); it != timings.end(); ++it)
  {
    const std::string task    = it->first;
    const uint num_timings    = it->second.first;
    const double total_time   = it->second.second;
    const double average_time = total_time / static_cast<double>(num_timings);

    table(task, "Average time") = average_time;
    table(task, "Total time")   = total_time;
    table(task, "Reps")         = num_timings;
  }
  table.print();

  // Clear timings
  if (reset)
    timings.clear();
}
//-----------------------------------------------------------------------------
double Logger::timing(std::string task, bool reset)
{
  // Find timing
  map_iterator it = timings.find(task);
  if (it == timings.end())
  {
    std::stringstream line;
    line << "No timings registered for task \"" << task << "\".";
    error(line.str());
  }

  // Compute average
  const uint num_timings  = it->second.first;
  const double total_time   = it->second.second;
  const double average_time = total_time / static_cast<double>(num_timings);

  // Clear timing
  timings.erase(it);

  return average_time;
}
//-----------------------------------------------------------------------------
void Logger::__debug(std::string msg) const
{
  std::string s = std::string("Debug: ") + msg;
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::__assert(std::string msg) const
{
  std::string s = std::string("*** Assertion ") + msg;
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::write(int debug_level, std::string msg) const
{
  // Check debug level
  if (debug_level > this->debug_level)
    return;

  // Prefix with process number if running in parallel
  if (process_number >= 0)
  {
    std::stringstream prefix;
    prefix << "Process " << process_number << ": ";
    msg = prefix.str() + msg;
  }

  // Add indentation
  for (int i = 0; i < indentation_level; i++)
    msg = "  " + msg;

  // Choose destination
  switch (destination)
  {
  case silent:
    // Do nothing if destination == silent
    do {} while (false);
    break;
  default:
    *logstream << msg << std::endl;
  }
}
//----------------------------------------------------------------------------
