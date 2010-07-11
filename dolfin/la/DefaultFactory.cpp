// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-17
// Last changed: 2009-09-08

#include <dolfin/parameter/GlobalParameters.h>
#include "uBLASFactory.h"
#include "PETScFactory.h"
#include "EpetraFactory.h"
#include "MTL4Factory.h"
#include "STLFactory.h"
#include "DefaultFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericMatrix* DefaultFactory::create_matrix() const
{
  return factory().create_matrix();
}
//-----------------------------------------------------------------------------
GenericVector* DefaultFactory::create_vector() const
{
  return factory().create_vector();
}
//-----------------------------------------------------------------------------
GenericVector* DefaultFactory::create_local_vector() const
{
  return factory().create_local_vector();
}
//-----------------------------------------------------------------------------
GenericSparsityPattern* DefaultFactory::create_pattern() const
{
  return factory().create_pattern();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& DefaultFactory::factory() const
{
  // Fallback
  const std::string default_backend = "uBLAS";
  typedef uBLASFactory<> DefaultFactory;

  // Get backend from parameter system
  const std::string backend = parameters["linear_algebra_backend"];

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
  info(WARNING, "Linear algebra backend \"" + backend + "\" not available, using " + default_backend + ".");
  return DefaultFactory::instance();
}
//-----------------------------------------------------------------------------
