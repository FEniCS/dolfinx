// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2007-05-14

#ifndef __LOGGER_H
#define __LOGGER_H

#include <string>
#include <ostream>
#include <dolfin/constants.h>

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
    void message(std::string msg, int debug_level = 0);

    /// Print warning
    void warning(std::string msg);

    /// Print error message and throw exception
    void error(std::string msg);

    /// Begin task (increase indentation level)
    void begin(std::string msg, int debug_level = 0);

    /// End task (decrease indentation level)
    void end();

    /// Draw progress bar
    void progress (std::string title, real p);

    /// Set output destination ("terminal" or "silent")
    void setOutputDestination(std::string destination);

    /// Set output destination to stream
    void setOutputDestination(std::ostream& stream);

    /// Set debug level
    void setDebugLevel(int debug_level);

    /// Set debug level
    inline int getDebugLevel() { return debug_level; }

    /// Helper function for dolfin_debug macro
    void __debug(std::string msg);

    /// Helper function for dolfin_assert macro
    void __assert(std::string msg);

  private:

    // Output destination
    enum Destination {terminal, stream, silent};
    Destination destination;

    // Write message to current output destination
    void write(int debug_level, std::string msg);
    
    // Current debug level
    int debug_level;

    // Current indentation level
    int indentation_level;

    std::ostream* logstream;

  };

}

#endif
