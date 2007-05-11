// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

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

    void info     (std::string msg);
    void info     (int debug_level, std::string msg);

    void warning(std::string msg, std::string location);
    void error(std::string msg, std::string location);
    void debug(std::string msg, std::string location);

    void dassert(std::string msg, std::string location);
    void progress (const char* title, const char* label, real p);

    void begin();
    void end();

    void active(bool state);
    void init(const char* destination);
    void level(int debug_level);

  private:

    // Output destination
    enum Destination {terminal, stream, silent};
    Destination destination;

    // Write message to current output destination
    void write(int debug_level, std::string msg);

    bool state;
    int debug_level;
    int indentation_level;

    char* buffer0;
    char* buffer1;

  };

}

#endif
