// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-02
// Last changed: 2005

#ifndef __SYNCHRONIZER_H
#define __SYNCHRONIZER_H 
 
#include <string>
#include <dolfin/dolfin_log.h>

namespace dolfin
{
  
  /// Associates an object with time t 

  class Synchronizer
  {
  public:
    
    /// Constructors
		Synchronizer();
    Synchronizer(const real& t);
    
    /// Destructor
    ~Synchronizer();

    /// Associate object with time t
		void sync(const real& t);
    
    /// Return the current time t
		real time() const;

  private:
    
    // Pointer to the current time
    const real* t;
    
    // True if synchronized to time
    bool time_set;

  };
  
}

#endif
