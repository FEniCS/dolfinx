// Copyright (C) 2003-2007 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-14
// Last changed: 2007-05-13

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <string>
#include <dolfin/constants.h>

namespace dolfin
{
  
  class Progress
  {
  public:
    
    Progress(std::string title, unsigned int n);
    Progress(std::string title);
    ~Progress();

    void setStep(real step);
    
    void operator=(unsigned int i);
    void operator=(real p);
    void operator++();
    void operator++(int);

    void stop();
    
    real value();
    std::string title();

  private:
    
    real checkBounds(unsigned int i);
    real checkBounds(real p);
    
    void update();
    
    std::string _title;
    
    real p0;
    real p1;

    real progress_step;
    
    unsigned int i;
    unsigned int n;

    bool stopped;
    
  };
  
}

#endif
