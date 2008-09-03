// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-17
// Last changed: 2008-08-07

#include <dolfin/parameter/parameters.h>
#include "uBLASFactory.h"
#include "PETScFactory.h"
#include "EpetraFactory.h"
#include "MTL4Factory.h"
#include "STLFactory.h"
#include "DefaultFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericMatrix* DefaultFactory::createMatrix() const
{
  return factory().createMatrix();
}
//-----------------------------------------------------------------------------
GenericVector* DefaultFactory::createVector() const
{
  return factory().createVector();
}
//-----------------------------------------------------------------------------
GenericSparsityPattern * DefaultFactory::createPattern() const
{
  return factory().createPattern();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& DefaultFactory::factory() const
{
  // Fallback
  std::string default_backend = "uBLAS";
  typedef uBLASFactory<> DefaultFactory;

  // Get backend from parameter system
  std::string backend = dolfin_get("linear algebra backend");

  // Choose backend
  if (backend == "uBLAS")
  {
    return uBLASFactory<>::instance();
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return PETScFactory::instance();
#else
    error("PETSc linear algebra backend is not available.");
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return EpetraFactory::instance();
#else
    error("Trilinos linear algebra backend is not available.");
#endif
  }
  else if (backend == "MTL4")
  {
#ifdef HAS_MTL4
    return MTL4Factory::instance();
#else
    error("MTL4 linear algebra backend is not available.");
#endif
  }
  else if (backend == "STL")
  { 
    return STLFactory::instance();
  }

  // Fallback
  message("Linear algebra backend \"" + backend + "\" not available, using " + default_backend + ".");
  return DefaultFactory::instance();
}
//-----------------------------------------------------------------------------
