// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2009-07-02

#ifndef __LOGGER_H
#define __LOGGER_H

#include <string>
#include <ostream>
#include <map>
#include <dolfin/common/types.h>
#include "LogLevel.h"

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
    void info(std::string msg, int log_level = INFO) const;

    /// Print underlined message
    void info_underline(std::string msg, int log_level = INFO) const;

    /// Print warning
    void warning(std::string msg) const;

    /// Print error message and throw exception
    void error(std::string msg) const;

    /// Begin task (increase indentation level)
    void begin(std::string msg, int log_level = PROGRESS);

    /// End task (decrease indentation level)
    void end();

    /// Draw progress bar
    void progress (std::string title, double p) const;

    /// Set output stream
    void set_output_stream(std::ostream& stream);

    /// Get output stream
    std::ostream& get_output_stream() { return *logstream; }

    /// Turn logging on or off
    void logging(bool active);

    /// Return true iff logging is active
    inline bool is_active() { return active; }

    /// Set log level
    void set_log_level(int log_level);

    /// Get log level
    inline int get_log_level() const { return log_level; }

    /// Register timing (for later summary)
    void register_timing(std::string task, double elapsed_time);

    /// Print summary of timings and tasks, optionally clearing stored timings
    void summary(bool reset=false);

    /// Return timing (average) for given task, optionally clearing timing for task
    double timing(std::string task, bool reset=false);

    /// Helper function for dolfin_debug macro
    void __debug(std::string msg) const;

  private:

    // Write message
    void write(int log_level, std::string msg) const;

    // True iff logging is active
    bool active;

    // Current log level
    int log_level;

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
