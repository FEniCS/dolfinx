// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug, 2007, 2009.
// Modified by Garth N. Wells, 2011.
//
// First added:  2003-03-13
// Last changed: 2016-06-10


#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#ifdef __linux__
#include <sys/types.h>
#include <unistd.h>
#endif

#include <dolfin/common/constants.h>
#include <dolfin/common/defines.h>
#include <dolfin/common/MPI.h>
#include <dolfin/io/File.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "LogLevel.h"
#include "Logger.h"

using namespace dolfin;

// Function for monitoring memory usage, called by thread
#ifdef __linux__
void _monitor_memory_usage(dolfin::Logger* logger)
{
  dolfin_assert(logger);

  // Open statm
  //std::fstream

  // Get process ID and page size
  const std::size_t pid = getpid();
  const size_t page_size = getpagesize();

  // Print some info
  std::stringstream s;
  s << "Initializing memory monitor for process " << pid << ".";
  logger->log(s.str());

  // Prepare statm file
  std::stringstream filename;
  filename << "/proc/" << pid << "/statm";
  std::ifstream statm;

  // Enter loop
  while (true)
  {
    // Sleep for a while
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Read number of pages from statm
    statm.open(filename.str().c_str());
    if (!statm)
      logger->error("Unable to open statm file for process.");
    size_t num_pages;
    statm >> num_pages;
    statm.close();

    // Convert to MB and report memory usage
    const size_t num_mb = num_pages*page_size / (1024*1024);
    logger->_report_memory_usage(num_mb);
  }
}
#endif

//-----------------------------------------------------------------------------
Logger::Logger() : _active(true), _log_level(INFO), _indentation_level(0),
                   logstream(&std::cout), _maximum_memory_usage(-1),
                   _mpi_comm(MPI_COMM_WORLD)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  // Join memory monitor thread if it exists
  if (_thread_monitor_memory_usage)
    _thread_monitor_memory_usage->join();
}
//-----------------------------------------------------------------------------
void Logger::log(std::string msg, int log_level) const
{
  write(log_level, msg);
}
//-----------------------------------------------------------------------------
void Logger::log_underline(std::string msg, int log_level) const
{
  if (msg.empty())
    log(msg, log_level);

  std::stringstream s;
  s << msg;
  s << "\n";
  for (int i = 0; i < _indentation_level; i++)
    s << "  ";
  for (std::size_t i = 0; i < msg.size(); i++)
    s << "-";

  log(s.str(), log_level);
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
void Logger::dolfin_error(std::string location,
                          std::string task,
                          std::string reason,
                          int mpi_rank) const
{

  if (mpi_rank < 0)
    mpi_rank = MPI::rank(_mpi_comm);
  std::string _mpi_rank = std::to_string(mpi_rank);

  std::stringstream s;
  s << std::endl << std::endl
    << "*** "
    << "-------------------------------------------------------------------------"
    << std::endl
    << "*** DOLFIN encountered an error. If you are not able to resolve this issue"
    << std::endl
    << "*** using the information listed below, you can ask for help at"
    << std::endl
    << "***" << std::endl
    << "***     fenics-support@googlegroups.com"
    << std::endl
    << "***" << std::endl
    << "*** Remember to include the error message listed below and, if possible,"
    << std::endl
    << "*** include a *minimal* running example to reproduce the error."
    << std::endl
    << "***" << std::endl
    << "*** "
    << "-------------------------------------------------------------------------"
    << std::endl
    << "*** " << "Error:   Unable to " << task << "." << std::endl
    << "*** " << "Reason:  " << reason << "." << std::endl
    << "*** " << "Where:   This error was encountered inside " << location << "."
    << std::endl
    << "*** " << "Process: " << _mpi_rank << std::endl
    << "*** " << std::endl
    << "*** " << "DOLFIN version: " << dolfin_version()  << std::endl
    << "*** " << "Git changeset:  " << git_commit_hash() << std::endl
    << "*** "
    << "-------------------------------------------------------------------------"
    << std::endl;

  throw std::runtime_error(s.str());
}
//-----------------------------------------------------------------------------
void Logger::deprecation(std::string feature,
                         std::string version_deprecated,
                         std::string message) const
{
  std::stringstream s;
  s << "*** "
    << "-------------------------------------------------------------------------"
    << std::endl
    << "*** Warning: " << feature << " has been deprecated in FEniCS version "
    << version_deprecated << "." << std::endl
    << "*** It will (likely) be removed in the next FEniCS release."
    << std::endl
    << "*** " << message << std::endl
    << "*** "
    << "-------------------------------------------------------------------------"
    << std::endl;

  #ifdef DOLFIN_DEPRECATION_ERROR
  error(s.str());
  #else
  write(WARNING, s.str());
  #endif
}
//-----------------------------------------------------------------------------
void Logger::begin(std::string msg, int log_level)
{
  // Write a message
  log(msg, log_level);
  _indentation_level++;
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  _indentation_level--;
}
//-----------------------------------------------------------------------------
void Logger::progress(std::string title, double p) const
{
  std::stringstream line;
  line << title << " [";

  const int N = DOLFIN_TERM_WIDTH - title.size() - 12 - 2*_indentation_level;
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
void Logger::set_log_active(bool active)
{
  _active = active;
}
//-----------------------------------------------------------------------------
void Logger::set_log_level(int log_level)
{
  _log_level = log_level;
}
//-----------------------------------------------------------------------------
void Logger::set_indentation_level(std::size_t indentation_level)
{
  _indentation_level = indentation_level;
}
//-----------------------------------------------------------------------------
void Logger::register_timing(std::string task,
                             std::tuple<double, double, double> elapsed)
{
  dolfin_assert(elapsed >=
    std::make_tuple(double(0.0), double(0.0), double(0.0)));

  // Print a message
  std::stringstream line;
  line << "Elapsed wall, usr, sys time: "
       << std::get<0>(elapsed) << ", "
       << std::get<1>(elapsed) << ", "
       << std::get<2>(elapsed)
       << " ("  << task << ")";
  log(line.str(), TRACE);

  // Store values for summary
  const auto timing = std::tuple_cat(std::make_tuple(std::size_t(1)), elapsed);
  auto it = _timings.find(task);
  if (it == _timings.end())
  {
    _timings[task] = timing;
  }
  else
  {
    std::get<0>(it->second) += std::get<0>(timing);
    std::get<1>(it->second) += std::get<1>(timing);
    std::get<2>(it->second) += std::get<2>(timing);
    std::get<3>(it->second) += std::get<3>(timing);
  }
}
//-----------------------------------------------------------------------------
void Logger::list_timings(TimingClear clear, std::set<TimingType> type)
{
  // Format and reduce to rank 0
  Table timings = this->timings(clear, type);
  timings = MPI::avg(_mpi_comm, timings);
  const std::string str = timings.str(true);

  // Print just on rank 0
  if (MPI::rank(_mpi_comm) == 0)
    log(str);

  // Print maximum memory usage if available
  if (_maximum_memory_usage >= 0)
  {
    std::stringstream s;
    s << "\nMaximum memory usage: " << _maximum_memory_usage << " MB";
    log(s.str());
  }
}
//-----------------------------------------------------------------------------
void Logger::dump_timings_to_xml(std::string filename, TimingClear clear)
{
  Table t = timings(clear,
    { TimingType::wall, TimingType::user, TimingType::system });

  Table t_max = MPI::max(_mpi_comm, t);
  Table t_min = MPI::min(_mpi_comm, t);
  Table t_avg = MPI::avg(_mpi_comm, t);

  if (MPI::rank(_mpi_comm) == 0)
  {
    File f(MPI_COMM_SELF, filename);
    f << t_max;
    f << t_min;
    f << t_avg;
  }
}
//-----------------------------------------------------------------------------
std::map<TimingType, std::string> Logger::_TimingType_descr
  = { { TimingType::wall,   "wall" },
      { TimingType::user,   "usr"  },
      { TimingType::system, "sys"  } };
//-----------------------------------------------------------------------------
Table Logger::timings(TimingClear clear,
                      std::set<TimingType> type)
{
  // Generate timing table
  Table table("Summary of timings");
  for (auto& it : _timings)
  {
    const std::string task = it.first;
    const std::size_t num_timings = std::get<0>(it.second);
    const std::vector<double> times { std::get<1>(it.second),
                                      std::get<2>(it.second),
                                      std::get<3>(it.second) };
    table(task, "reps") = num_timings;
    for (const auto& t : type)
    {
      const double total_time = times[static_cast<int>(t)];
      const double average_time = total_time / static_cast<double>(num_timings);
      table(task, Logger::_TimingType_descr[t] + " avg") = average_time;
      table(task, Logger::_TimingType_descr[t] + " tot") = total_time;
    }

  }

  // Clear timings
  if (static_cast<bool>(clear))
    _timings.clear();

  return table;
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double, double, double>
  Logger::timing(std::string task, TimingClear clear)
{
  // Find timing
  auto it = _timings.find(task);
  if (it == _timings.end())
  {
    std::stringstream line;
    line << "No timings registered for task \"" << task << "\".";
    dolfin_error("Logger.cpp",
                 "extract timing for task",
                 line.str());
  }
  // Prepare for return for the case of reset
  const auto result = it->second;

  // Clear timing
  if (static_cast<bool>(clear))
    _timings.erase(it);

  return result;
}
//-----------------------------------------------------------------------------
void Logger::monitor_memory_usage()
{
  #ifndef __linux__
  warning("Unable to initialize memory monitor; only available on GNU/Linux.");
  return;

  #else

  // Check that thread has not already been started
  if (_thread_monitor_memory_usage)
  {
    log("Memory monitor already initialize.");
    return;
  }

  // Create thread
  _thread_monitor_memory_usage.reset(new std::thread(std::bind(&_monitor_memory_usage, this)));

  #endif
}
//-----------------------------------------------------------------------------
void Logger::_report_memory_usage(size_t num_mb)
{
  std::stringstream s;
  s << "Memory usage: " << num_mb << " MB";
  log(s.str());
  _maximum_memory_usage = std::max(_maximum_memory_usage,
                                   static_cast<long int>(num_mb));
}
//-----------------------------------------------------------------------------
void Logger::__debug(std::string msg) const
{
  std::string s = std::string("DEBUG: ") + msg;
  write(DBG, s);
}
//-----------------------------------------------------------------------------
void Logger::__dolfin_assert(std::string file, unsigned long line,
                             std::string function, std::string check) const
{
  std::stringstream location;
  location << file << " (line " << line << ")";
  std::stringstream task;
  task << "complete call to function " << function << "()";
  std::stringstream reason;
  reason << "Assertion " << check << " failed";
  dolfin_error(location.str(), task.str(), reason.str());
}
//-----------------------------------------------------------------------------
void Logger::write(int log_level, std::string msg) const
{
  // Check log level
  if (!_active || log_level < _log_level)
    return;

  const std::size_t rank = MPI::rank(_mpi_comm);

  // Check if we want output on root process only
  const bool std_out_all_processes = parameters["std_out_all_processes"];
  if (rank > 0 && !std_out_all_processes && log_level < WARNING)
    return;

  // Prefix with process number if running in parallel
  if (MPI::size(_mpi_comm) > 1)
  {
    std::stringstream prefix;
    prefix << "Process " << rank << ": ";
    msg = prefix.str() + msg;
  }

  // Add indentation
  for (int i = 0; i < _indentation_level; i++)
    msg = "  " + msg;

  // Write to stream
  *logstream << msg << std::endl;
}
//----------------------------------------------------------------------------
