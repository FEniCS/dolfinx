// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2010-2012
//
// First added:  2010-02-25
// Last changed: 2012-04-11

#ifdef HAS_PETSC

#include <petscksp.h>
#include <petscmat.h>

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "PETScKrylovSolver.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include "PETScPreconditioner.h"

using namespace dolfin;

// Mapping from preconditioner string to PETSc
const std::map<std::string, const PCType> PETScPreconditioner::_methods
= { {"default",          ""},
    {"ilu",              PCILU},
    {"icc",              PCICC},
    {"jacobi",           PCJACOBI},
    {"bjacobi",          PCBJACOBI},
    {"sor",              PCSOR},
    {"additive_schwarz", PCASM},
    {"petsc_amg",        PCGAMG},
#if PETSC_HAVE_HYPRE
    {"hypre_amg",        PCHYPRE},
    {"hypre_euclid",     PCHYPRE},
    {"hypre_parasails",  PCHYPRE},
#endif
#if PETSC_HAVE_ML
    {"amg",              PCML},
    {"ml_amg",           PCML},
#elif PETSC_HAVE_HYPRE
    {"amg",              PCHYPRE},
#endif
    {"none",             PCNONE} };

// Mapping from preconditioner string to description string
const std::map<std::string, std::string>
PETScPreconditioner::_methods_descr
= { {"default",          "default preconditioner"},
    {"ilu",              "Incomplete LU factorization"},
    {"icc",              "Incomplete Cholesky factorization"},
    {"sor",              "Successive over-relaxation"},
    {"petsc_amg",        "PETSc algebraic multigrid"},
#if PETSC_HAVE_HYPRE
    {"amg",              "Algebraic multigrid"},
    {"hypre_amg",        "Hypre algebraic multigrid (BoomerAMG)"},
    {"hypre_euclid",     "Hypre parallel incomplete LU factorization"},
    {"hypre_parasails",  "Hypre parallel sparse approximate inverse"},
#endif
#if PETSC_HAVE_ML
    {"ml_amg",           "ML algebraic multigrid"},
#endif
    {"none",             "No preconditioner"} };
//-----------------------------------------------------------------------------
std::map<std::string, std::string>
PETScPreconditioner::preconditioners()
{
  return PETScPreconditioner::_methods_descr;
}
//-----------------------------------------------------------------------------
Parameters PETScPreconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters()("preconditioner"));
  p.rename("petsc_preconditioner");

  // General parameters
  //p.add("view", false);

  // Prefix for setting options
  p.add("options_prefix", "default");

  return p;
}
//-----------------------------------------------------------------------------
PETScPreconditioner::PETScPreconditioner(std::string type)
  : _type(type), gdim(0)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that the requested method is known
  if (_methods.count(type) == 0)
  {
    dolfin_error("PETScPreconditioner.cpp",
                 "create PETSc preconditioner",
                 "Unknown preconditioner type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
PETScPreconditioner::~PETScPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set(PETScKrylovSolver& solver)
{
  PetscErrorCode ierr;
  dolfin_assert(solver.ksp());

  // Get PETSc PC pointer
  PC pc;
  ierr = KSPGetPC(solver.ksp(), &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Treat special cases  first
  if (_type.find("hypre") != std::string::npos)
  {
    #if PETSC_HAVE_HYPRE
    ierr = PCSetType(pc, PCHYPRE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    if (_type == "hypre_amg" || _type == "amg")
    {
      ierr = PCHYPRESetType(pc, "boomeramg");
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPRESetType");
    }
    else if (_type == "hypre_euclid")
    {
      ierr = PCHYPRESetType(pc, "euclid");
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPRESetType");
    }
    else
    {
      dolfin_error("PETScPreconditioner.cpp",
                   "set PETSc preconditioner",
                   "Requested Hypre preconditioner unknown (note that pilut is not supported)");
    }
    #else
    warning("PETSc has not been compiled with the HYPRE library for "
            "algebraic multigrid. Default PETSc solver will be used. "
            "For performance, installation of HYPRE is recommended.");
    #endif
  }
  else if (_type == "amg_ml" || _type == "ml_amg")
  {
    #if PETSC_HAVE_ML

    // Set preconditioner to ML
    ierr = PCSetType(pc, PCML);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    #else
    warning("PETSc has not been compiled with the ML library for   "
            "algebraic multigrid. Default PETSc solver will be used. "
            "For performance, installation of ML is recommended.");
    #endif
  }
  else if (_type == "petsc_amg")
  {
    // Set preconditioner to GAMG
    ierr = PCSetType(pc, PCGAMG);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
  }
  else if (_type == "additive_schwarz")
  {
    // Select method and overlap
    ierr = PCSetType(pc, _methods.find("additive_schwarz")->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    const int schwarz_overlap = parameters("schwarz")["overlap"];
    PCASMSetOverlap(pc, schwarz_overlap);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCASMSetOverlap");
  }
  else if (_type != "default")
  {
    ierr = PCSetType(pc, _methods.find(_type)->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
    ierr = PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftType");
    ierr = PCFactorSetShiftAmount(pc, parameters["shift_nonzero"]);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftAmount");
  }

  const int ilu_levels = parameters("ilu")["fill_level"];
  ierr = PCFactorSetLevels(pc, ilu_levels);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetLevels");

  // Set physical coordinates for row dofs
  if (!_coordinates.empty())
  {
    dolfin_assert(gdim > 0);
    ierr = PCSetCoordinates(pc, gdim, _coordinates.size()/gdim,
                            _coordinates.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetCoordinates");
  }

  // Clear memory
  _coordinates.clear();

  std::string prefix = std::string(parameters["options_prefix"]);
  if (prefix != "default")
  {
    // Make sure that the prefix has a '_' at the end if the user
    // didn't provide it
    char lastchar = *prefix.rbegin();
    if (lastchar != '_')
      prefix += "_";

    PCSetOptionsPrefix(pc, prefix.c_str());
  }
  PCSetFromOptions(pc);

  // Print preconditioner information
  /*
  const bool report = parameters["report"];
  if (report)
  {
    ierr = PCSetUp(pc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetUp");
    ierr = PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCView");
  }
  */
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_coordinates(const std::vector<double>& x,
                                          std::size_t dim)
{
  _coordinates = x;
  gdim = dim;
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_fieldsplit(
  PETScKrylovSolver& solver,
  const std::vector<std::vector<dolfin::la_index>>& fields,
  const std::vector<std::string>& split_names)
{
  dolfin_assert(fields.size() == split_names.size());
  PetscErrorCode ierr;

  if (fields.empty())
    return;

  // Get PETSc PC pointer
  PC pc;
  dolfin_assert(solver.ksp());
  ierr = KSPGetPC(solver.ksp(), &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Add split for each field
  for (std::size_t i = 0; i < fields.size(); ++i)
  {
    // Create IndexSet
    IS is;
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, fields[i].size(), fields[i].data(),
                           PETSC_USE_POINTER, &is);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");

    // Add split
    ierr = PCFieldSplitSetIS(pc, split_names[i].c_str(), is);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCFieldSplitSetIS");

    // Clean up IndexSet
    ierr = ISDestroy(&is);
    if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
  }
}
//-----------------------------------------------------------------------------
std::string PETScPreconditioner::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
    warning("Verbose output for PETScPreconditioner not implemented.");
  else
    s << "<PETScPreconditioner>";

  return s.str();
}
//-----------------------------------------------------------------------------

#endif
