// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#ifndef __TERMINAL_LOGGER_H
#define __TERMINAL_LOGGER_H

#include <dolfin/constants.h>
#include <dolfin/GenericLogger.h>

namespace dolfin
{

  class TerminalLogger : public GenericLogger
  {
  public:
	 
    TerminalLogger();
    ~TerminalLogger();
    
    void info     (const char* msg);
    void debug    (const char* msg, const char* location);
    void warning  (const char* msg, const char* location);
    void error    (const char* msg, const char* location);
    void dassert  (const char* msg, const char* location);
    void progress (const char* title, const char* label, real p);
    
  private:

    void indent();
    
  };

}
  
#endif
