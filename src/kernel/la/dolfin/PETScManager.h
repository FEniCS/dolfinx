// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PETSC_MANAGER_H
#define __PETSC_MANAGER_H

namespace dolfin
{
  /// This class is responsible for initializing and (automatically)
  /// finalizing PETSc. To initialize PETSc, call PETScManager::init()
  /// once (additional calls will be ignored). Finalization will be
  /// handled automatically.

  class PETScManager
  {
  public:

    /// Initialize PETSc
    static void init();
    
  protected:

    // Constructor
    PETScManager();

    // Destructor
    ~PETScManager();

  private:

    // Singleton instance
    static PETScManager petsc;
    
  };
  
}

#endif
