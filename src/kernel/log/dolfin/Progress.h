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

	 Progress(const char* title, int n);
	 Progress(const char* title);
	 ~Progress();

	 void operator=(int i);
	 void operator=(real p);
	 void operator++();
	 void operator++(int);

	 void update(int i,  const char* format, ...);
	 void update(real p, const char* format, ...);
	 
  private:

	 real checkBounds(int i);
	 real checkBounds(real p);
	 
	 void update();
	 
	 char* title;
	 char* label;
	 
	 real p0;
	 real p1;
	 
	 int i;
	 int n;
	 
  };

}

#endif
