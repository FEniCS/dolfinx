// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_LOGGER_H
#define __GENERIC_LOGGER_H

#include <dolfin/Progress.h>
#include <dolfin/constants.h>

namespace dolfin {

  class GenericLogger {
  public:
	 
	 GenericLogger();
	 virtual ~GenericLogger() {};

	 virtual void info    (const char* msg) = 0;
	 virtual void debug   (const char* msg, const char* location) = 0;
	 virtual void warning (const char* msg, const char* location) = 0;
	 virtual void error   (const char* msg, const char* location) = 0;
	 virtual void dassert (const char* msg, const char* location) = 0;
	 virtual void progress(const char* title, const char* label, real p) = 0;

	 virtual void update() = 0;
	 virtual void quit() = 0;
	 virtual bool finished() = 0;
	 
	 virtual void progress_add    (Progress* p) = 0;
	 virtual void progress_remove (Progress *p) = 0;
	 
	 void start();
	 void end();

  private:

	 int level;
	 
  };

}
  
#endif
