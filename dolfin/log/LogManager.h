// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#ifndef __LOG_MANAGER_H
#define __LOG_MANAGER_H

#include "Logger.h"

namespace dolfin
{
  
  class LogManager
  {
  public:

    // Singleton instance of logger
    static Logger logger;
	 
  };

}
  
#endif
