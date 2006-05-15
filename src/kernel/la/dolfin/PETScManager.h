// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-12-09
// Last changed: 2006-05-15

#ifndef __PETSC_MANAGER_H
#define __PETSC_MANAGER_H

#ifdef HAVE_PETSC_H

// Due to a bug in PETSC, the Boost headers must be included
// before including pescerror.h.

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <petscconf.h>
#include <petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>

namespace dolfin
{
  /// This class is responsible for initializing and (automatically)
  /// finalizing PETSc. To initialize PETSc, call PETScManager::init()
  /// once (additional calls will be ignored). Finalization will be
  /// handled automatically.

  class PETScManager
  {
  public:

    /// Initialize PETSc without command-line arguments
    static void init();

    /// Initialize PETSc with command-line arguments
    static void init(int argc, char* argv[]);
    
  protected:

    // Constructor
    PETScManager();

    // Destructor
    ~PETScManager();

  private:

    // Singleton instance
    static PETScManager petsc;

    // Check if initialized
    bool initialized;
    
  };
  
}

#endif

#endif
