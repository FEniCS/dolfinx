// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2008-09-18

#ifndef __LOGGER_H
#define __LOGGER_H

#include <string>
#include <ostream>
#include <map>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Logger
  {
  public:
    
    /// Constructor
    Logger();

    /// Destructor
    ~Logger();
    
    /// Print message
    void message(std::string msg, int debug_level = 0) const;

    /// Print warning
    void warning(std::string msg) const;

    /// Print error message and throw exception
    void error(std::string msg) const;

    /// Begin task (increase indentation level)
    void begin(std::string msg, int debug_level = 0);

    /// End task (decrease indentation level)
    void end();

    /// Draw progress bar
    void progress (std::string title, double p) const;

    /// Set output destination ("terminal" or "silent")
    void setOutputDestination(std::string destination);

    /// Set output destination to stream
    void setOutputDestination(std::ostream& stream);

    /// Set debug level
    void setDebugLevel(int debug_level);

    /// Set debug level
    inline int getDebugLevel() const { return debug_level; }

    /// Register timing (for later summary)
    void registerTiming(std::string task, double elapsed_time);

    /// Print summary of timings and tasks, optionally clearing stored timings
    void summary(bool reset=false);
    
    /// Return timing (average) for given task, optionally clearing timing for task
    double timing(std::string task, bool reset=false);

    /// Helper function for dolfin_debug macro
    void __debug(std::string msg) const;

    /// Helper function for dolfin_assert macro
    void __assert(std::string msg) const;

  private:

    // Output destination
    enum Destination {terminal, stream, silent};
    Destination destination;

    // Write message to current output destination
    void write(int debug_level, std::string msg) const;
    
    // Current debug level
    int debug_level;

    // Current indentation level
    int indentation_level;

    // Optional stream for logging
    std::ostream* logstream;

    // List of timings for tasks, map from string to (num_timings, total_time)
    std::map<std::string, std::pair<uint, double> > timings;

    // Process number (-1 if we are not running in parallel)
    int process_number;

  };

}

#endif
