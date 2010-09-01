// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2010-09-01

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
  : active(true), log_level(PROGRESS), indentation_level(0), logstream(&std::cout),
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
void Logger::info(std::string msg, int log_level) const
{
  write(log_level, msg);
}
//-----------------------------------------------------------------------------
void Logger::info_underline(std::string msg, int log_level) const
{
  if (msg.size() == 0)
    info(msg, log_level);

  std::stringstream s;
  s << msg;
  s << "\n";
  for (int i = 0; i < indentation_level; i++)
    s << "  ";
  for (uint i = 0; i < msg.size(); i++)
    s << "-";

  info(s.str(), log_level);
}
//-----------------------------------------------------------------------------
void Logger::warning(std::string msg) const
{
  std::string s = std::string("*** Warning: ") + msg;
  write(WARNING, s);
}
//-----------------------------------------------------------------------------
void Logger::error(std::string msg) const
{
  std::string s = std::string("*** Error: ") + msg;
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::begin(std::string msg, int log_level)
{
  // Write a message
  info(msg, log_level);
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
  std::stringstream line;
  line << title << " [";

  const int N = DOLFIN_TERM_WIDTH - title.size() - 12 - 2*indentation_level;
  const int n = static_cast<int>(p*static_cast<double>(N));

  for (int i = 0; i < n; i++)
    line << '=';
  if (n < N)
    line << '>';
  for (int i = n+1; i < N; i++)
    line << ' ';

  line << std::setiosflags(std::ios::fixed);
  line << std::setprecision(1);
  line << "] " << 100.0*p << '%';

  write(PROGRESS, line.str());
}
//-----------------------------------------------------------------------------
void Logger::set_output_stream(std::ostream& ostream)
{
  logstream = &ostream;
}
//-----------------------------------------------------------------------------
void Logger::logging(bool active)
{
  this->active = active;
}
//-----------------------------------------------------------------------------
void Logger::set_log_level(int log_level)
{
  this->log_level = log_level;
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
  info(line.str(), TRACE);

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
  info(table.str(true));

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
  write(DBG, s);
}
//-----------------------------------------------------------------------------
void Logger::write(int log_level, std::string msg) const
{
  // Check log level
  if (!active || log_level < this->log_level)
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

  // Write to stream
  *logstream << msg << std::endl;
}
//----------------------------------------------------------------------------
