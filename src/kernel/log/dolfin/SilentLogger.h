// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SILENT_LOGGER_H
#define __SILENT_LOGGER_H

#include <dolfin/constants.h>
#include <dolfin/GenericLogger.h>

namespace dolfin {

  class SilentLogger : public GenericLogger {
  public:
	 
    SilentLogger();
    ~SilentLogger();
    
    void info    (const char* msg);
    void debug   (const char* msg, const char* location);
    void warning (const char* msg, const char* location);
    void error   (const char* msg, const char* location);
    void dassert (const char* msg, const char* location);
    void progress(const char* title, const char* label, real p);
    
    void update();
    void quit();
    bool finished();
    
    void progress_add    (Progress* p);
    void progress_remove (Progress *p);
    
  };

}
  
#endif
