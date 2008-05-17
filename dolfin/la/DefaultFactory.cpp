// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-17
// Last changed: 2008-05-17

#include <dolfin/parameter/parameters.h>
#include "uBlasFactory.h"
#include "PETScFactory.h"
#include "EpetraFactory.h"
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
  typedef uBlasFactory DefaultFactory;

  // Get backend from parameter system
  std::string backend = dolfin_get("linear algebra backend");

  // Choose backend
  if (backend == "uBLAS")
  {
    return uBlasFactory::instance();
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return PETScFactory::instance();
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return EpetraFactory::instance();
#endif
  }

  // Fallback
  message("Linear algebra backend \"" + backend + "\" not available, using " + default_backend + ".");
  return DefaultFactory::instance();
}
//-----------------------------------------------------------------------------
