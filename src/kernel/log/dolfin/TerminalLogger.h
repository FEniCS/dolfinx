// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TERMINAL_LOGGER_H
#define __TERMINAL_LOGGER_H

#include <dolfin/GenericLogger.h>

namespace dolfin {

  class TerminalLogger : public GenericLogger {
  public:
	 
	 TerminalLogger();
	 ~TerminalLogger();
	 
	 void info    (const char* msg);
	 void debug   (const char* msg);
	 void warning (const char* msg);
	 void error   (const char* msg);
	 
  };

}
  
#endif
