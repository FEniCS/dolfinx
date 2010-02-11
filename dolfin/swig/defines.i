/* -*- C -*- */
// Copyright (C) 2009 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-03
// Last changed: 2010-01-15

// ===========================================================================
// SWIG directives for mapping defines to Python
// ===========================================================================

%inline %{
bool has_mpi()
{
#ifdef HAS_MPI
  return true;
#else
  return false;
#endif
}

bool has_slepc()
{
#ifdef HAS_SLEPC
  return true;
#else
  return false;
#endif
}

bool has_scotch()
{
#ifdef HAS_SCOTCH
  return true;
#else
  return false;
#endif
}

bool has_cgal()
{
#ifdef HAS_CGAL
  return true;
#else
  return false;
#endif
}

bool has_umfpack()
{
#ifdef HAS_UMFPACK
  return true;
#else
  return false;
#endif
}

bool has_cholmod()
{
#ifdef HAS_CHOLMOD
  return true;
#else
  return false;
#endif
}

bool has_parmetis()
{
#ifdef HAS_PARMETIS
  return true;
#else
  return false;
#endif
}

bool has_gmp()
{
#ifdef HAS_GMP
  return true;
#else
  return false;
#endif
}

bool has_zlib()
{
#ifdef HAS_ZLIB
  return true;
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------
// Define a function that return true; if a specific la backend is supported
// ---------------------------------------------------------------------------
bool has_la_backend(std::string backend)
{
  if (backend == "uBLAS")
  {
    return true;
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "MTL4")
  {
#ifdef HAS_MTL4
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "STL")
  {
    return true;
  }
  return false;
}

%}

%feature("docstring") has_linear_algebra_backend "
Returns True if a linear algebra backend is available.
";

