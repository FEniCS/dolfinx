// Copyright (C) 2003 Jim Tilander.
// Licensed under the GNU GPL Version 2.
//
// Modified for DOLFIN by Anders Logg.

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <stdarg.h>
#include <dolfin/constants.h>

namespace dolfin {
  
  class Progress {
  public:
    
    Progress(const char* title, unsigned int n);
    Progress(const char* title);
    ~Progress();
    
    void operator=(unsigned int i);
    void operator=(real p);
    void operator++();
    void operator++(int);
    
    void update(unsigned int i,  const char* format, ...);
    void update(real p, const char* format, ...);
    
    real value();
    const char* title();
    const char* label();
    
  private:
    
    real checkBounds(unsigned int i);
    real checkBounds(real p);
    
    void update();
    
    char* _title;
    char* _label;
    
    real p0;
    real p1;

    real progress_step;
    
    unsigned int i;
    unsigned int n;
    
  };
  
}

#endif
