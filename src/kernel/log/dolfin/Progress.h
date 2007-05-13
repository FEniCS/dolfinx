// Copyright (C) 2003-2007 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-14
// Last changed: 2007-05-13

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <stdarg.h>
#include <dolfin/constants.h>

namespace dolfin
{
  
  class Progress
  {
  public:
    
    Progress(const char* title, unsigned int n);
    Progress(const char* title);
    ~Progress();

    void setStep(real step);
    
    void operator=(unsigned int i);
    void operator=(real p);
    void operator++();
    void operator++(int);

    void stop();
    
    real value();
    const char* title();

  private:
    
    real checkBounds(unsigned int i);
    real checkBounds(real p);
    
    void update();
    
    char* _title;
    
    real p0;
    real p1;

    real progress_step;
    
    unsigned int i;
    unsigned int n;

    bool stopped;
    
  };
  
}

#endif
