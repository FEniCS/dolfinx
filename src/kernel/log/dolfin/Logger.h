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
#include <dolfin/constants.h>

namespace dolfin
{

  class Logger
  {
  public:
    
    Logger();
    ~Logger();

    void info     (std::string msg, int debug_level = 0);

    void warning  (std::string msg);
    void error    (std::string msg);

    void debug    (std::string msg, std::string location);

    void dassert  (std::string msg, std::string location);
    void progress (std::string title, real p);

    void begin(std::string msg, int debug_level = 0);
    void end();

    void init     (std::string destination);
    void level    (int debug_level);

  private:

    // Output destination
    enum Destination {terminal, stream, silent};
    Destination destination;

    // Write message to current output destination
    void write(int debug_level, std::string msg);

    int debug_level;
    int indentation_level;

  };

}

#endif
