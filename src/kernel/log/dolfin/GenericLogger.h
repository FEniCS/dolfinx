// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_LOGGER_H
#define __GENERIC_LOGGER_H

namespace dolfin {

  class GenericLogger {
  public:
	 
	 GenericLogger();
	 virtual ~GenericLogger() {};

	 virtual void info    (const char* msg) = 0;
	 virtual void debug   (const char* msg) = 0;
	 virtual void warning (const char* msg) = 0;
	 virtual void error   (const char* msg) = 0;

	 void start();
	 void end();

  private:

	 int level;
	 
  };

}
  
#endif
