// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#ifndef __GENERIC_LOGGER_H
#define __GENERIC_LOGGER_H

#include <dolfin/constants.h>

namespace dolfin
{

  class Progress;

  class GenericLogger
  {
  public:
    
    GenericLogger();
    virtual ~GenericLogger() {};
    
    virtual void info     (const char* msg) = 0;
    virtual void debug    (const char* msg, const char* location) = 0;
    virtual void warning  (const char* msg, const char* location) = 0;
    virtual void error    (const char* msg, const char* location) = 0;
    virtual void dassert  (const char* msg, const char* location) = 0;
    virtual void progress (const char* title, const char* label, real p) = 0;
    
    void begin();
    void end();
    
  protected:
    
    int level;
    
  };
  
}

#endif
