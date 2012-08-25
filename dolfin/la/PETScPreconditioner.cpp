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

#include <boost/assign/list_of.hpp>
#include <boost/lexical_cast.hpp>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpcmg.h>
#include <dolfin/common/MPI.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScPreconditioner.h"

using namespace dolfin;

// Mapping from preconditioner string to PETSc
const std::map<std::string, const PCType> PETScPreconditioner::_methods
  = boost::assign::map_list_of("default",          "")
                              ("none",             PCNONE)
                              ("ilu",              PCILU)
                              ("icc",              PCICC)
                              ("jacobi",           PCJACOBI)
                              ("bjacobi",          PCBJACOBI)
                              ("sor",              PCSOR)
                              ("additive_schwarz", PCASM)
                              #if PETSC_HAVE_HYPRE
                              ("hypre_amg",        PCHYPRE)
                              ("hypre_euclid",     PCHYPRE)
                              ("hypre_parasails",  PCHYPRE)
                              #endif
                              #if PETSC_HAVE_ML
                              ("amg",              PCML)
                              ("ml_amg",           PCML)
                              #elif PETSC_HAVE_HYPRE
                              ("amg",              PCHYPRE)
                              #endif
                              ;

// Mapping from preconditioner string to description string
const std::vector<std::pair<std::string, std::string> > PETScPreconditioner::_methods_descr
  = boost::assign::pair_list_of
    ("default",          "default preconditioner")
    ("none",             "No preconditioner")
    ("ilu",              "Incomplete LU factorization")
    ("icc",              "Incomplete Cholesky factorization")
    ("sor",              "Successive over-relaxation")
    #if HAS_PETSC_CUSP
    ("jacobi",           "Jacobi iteration (GPU enabled)")
    ("bjacobi",          "Block Jacobi iteration (GPU enabled)")
    ("additive_schwarz", "Additive Schwarz (GPU enabled)")
    #else
    ("jacobi",           "Jacobi iteration")
    ("bjacobi",          "Block Jacobi iteration")
    ("additive_schwarz", "Additive Schwarz")
    #endif
    #if PETSC_HAVE_HYPRE
    ("amg",              "Algebraic multigrid")
    ("hypre_amg",        "Hypre algebraic multigrid (BoomerAMG)")
    ("hypre_euclid",     "Hypre parallel incomplete LU factorization")
    ("hypre_parasails",  "Hypre parallel sparse approximate inverse")
    #endif
    #if PETSC_HAVE_ML
    ("ml_amg",           "ML algebraic multigrid")
    #endif
    ;
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScPreconditioner::preconditioners()
{
  return PETScPreconditioner::_methods_descr;
}
//-----------------------------------------------------------------------------
Parameters PETScPreconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters()("preconditioner"));
  p.rename("petsc_preconditioner");

  // Hypre/ParaSails parameters
  Parameters p_parasails("parasails");
  p_parasails.add<double>("threshold");
  p_parasails.add<uint>("levels");

  // Hypre/BoomerAMG parameters
  Parameters p_boomeramg("BoomerAMG");
  p_boomeramg.add<uint>("num_sweeps");
  p_boomeramg.add<std::string>("cycle_type");             // "V" or "W"
  p_boomeramg.add<uint>("max_levels");
  p_boomeramg.add<double>("strong_threshold");
  p_boomeramg.add<double>("relaxation_weight");
  p_boomeramg.add<uint>("agressive_coarsening_levels");

  // Hypre package parameters
  Parameters p_hypre("hypre");
  p_hypre.add(p_parasails);
  p_hypre.add(p_boomeramg);
  p.add(p_hypre);

  return p;
}
//-----------------------------------------------------------------------------
PETScPreconditioner::PETScPreconditioner(std::string type) : type(type)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that the requested method is known
  if (_methods.count(type) == 0)
  {
    dolfin_error("PETScPreconditioner.cpp",
                 "create PETSc preconditioner",
                 "Unknown norm type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
PETScPreconditioner::~PETScPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set(PETScKrylovSolver& solver) const
{
  dolfin_assert(solver.ksp());

  // Get PETSc PC pointer
  PC pc;
  KSPGetPC(*(solver.ksp()), &pc);

  // Treat special cases  first
  if (type.find("hypre") != std::string::npos)
  {
    #if PETSC_HAVE_HYPRE
    PCSetType(pc, PCHYPRE);

    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    PCFactorSetShiftAmount(pc, PETSC_DECIDE);
    #else
    PCFactorSetShiftNonzero(pc, PETSC_DECIDE);
    #endif

    if (type == "hypre_amg" || type == "amg")
    {
      PCHYPRESetType(pc, "boomeramg");
      if (parameters("hypre")("BoomerAMG")["num_sweeps"].is_set())
      {
        const uint num_sweeps = parameters("hypre")("BoomerAMG")["num_sweeps"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_grid_sweeps_all",
                         boost::lexical_cast<std::string>(num_sweeps).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["cycle_type"].is_set())
      {
        const std::string cycle = parameters("hypre")("BoomerAMG")["cycle_type"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_cycle_type", cycle.c_str());
      }
      if (parameters("hypre")("BoomerAMG")["max_levels"].is_set())
      {
        const uint max_levels = parameters("hypre")("BoomerAMG")["max_levels"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_max_levels",
                          boost::lexical_cast<std::string>(max_levels).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["strong_threshold"].is_set())
      {
        const double threshold = parameters("hypre")("BoomerAMG")["strong_threshold"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold",
                          boost::lexical_cast<std::string>(threshold).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["relaxation_weight"].is_set())
      {
        const double relax = parameters("hypre")("BoomerAMG")["strong_theshold"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_relax_weight_all",
                            boost::lexical_cast<std::string>(relax).c_str() );
      }
      if (parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"].is_set())
      {
        const uint levels = parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_agg_nl",
                            boost::lexical_cast<std::string>(levels).c_str() );
      }
    }
    else if (type == "hypre_parasails")
    {
      PCHYPRESetType(pc, "parasails");
      if (parameters("hypre")("parasails")["threshold"].is_set())
      {
        const double thresh = parameters("hypre")("parasails")["threshold"];
        PetscOptionsSetValue("-pc_hypre_parasails_thresh", boost::lexical_cast<std::string>(thresh).c_str());
      }
      if (parameters("hypre")("parasails")["levels"].is_set())
      {
        const int levels = parameters("hypre")("parasails")["levels"];
        PetscOptionsSetValue("-pc_hypre_parasails_nlevels", boost::lexical_cast<std::string>(levels).c_str());
      }
    }
    else if (type == "hypre_euclid")
    {
      PCHYPRESetType(pc, "euclid");
      const uint ilu_level = parameters("ilu")["fill_level"];
      PetscOptionsSetValue("-pc_hypre_euclid_levels",
                          boost::lexical_cast<std::string>(ilu_level).c_str());
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
  else if (type == "amg_ml" || type == "ml_amg")
  {
    #if PETSC_HAVE_ML

    PCSetType(pc, PCML);

    //PCMGSetLevels(pc, 3, PETSC_NULL);
    //PCMGSetCycleType(pc, PC_MG_CYCLE_W);

    // -- PETSC general multigrid options

    // multiplicative or additive
    //PetscOptionsSetValue("-pc_mg_type", "additive");

    //PetscOptionsSetValue("-mg_levels",
    //                      boost::lexical_cast<std::string>(3).c_str());

    //PetscOptionsSetValue("-pc_mg_smoothup",
    //                      boost::lexical_cast<std::string>(1).c_str());
    //PetscOptionsSetValue("-pc_mg_smoothdown",
    //                      boost::lexical_cast<std::string>(1).c_str());

    //PetscOptionsSetValue("-pc_mg_cycles",
    //                      boost::lexical_cast<std::string>(1).c_str());

    // Chebychev
    //PetscOptionsSetValue("-mg_levels_ksp_type", "chebyshev");

    //PetscOptionsSetValue("-mg_levels_ksp_type", "richardson");
    //PetscOptionsSetValue("-mg_levels_ksp_initial_guess_nonzero", "1");

    //PetscOptionsSetValue("mg_levels_ksp_chebyshev_estimate_eigenvalues",
    //                      "0.0,1.1");

    //PetscOptionsSetValue("-mg_levels_ksp_max_it",
    //                      boost::lexical_cast<std::string>(2).c_str());

    // Smoother preconditioner
    //PetscOptionsSetValue("-mg_levels_pc_type", "none");
    //PetscOptionsSetValue("-mg_levels_pc_type", "jacobi");
    //PetscOptionsSetValue("-mg_levels_pc_type", "sor");

    //PetscOptionsSetValue("-mg_levels_pc_sor_its",
    //                      boost::lexical_cast<std::string>(2).c_str());

    //PetscOptionsSetValue("-mg_levels_pc_sor_lits",
    //                      boost::lexical_cast<std::string>(6).c_str());

    //PetscOptionsSetValue("-mg_levels_pc_sor_omega",
    //                      boost::lexical_cast<std::string>(0.8).c_str());

    //PetscOptionsSetValue("-pc_mg_monitor",
    //                      boost::lexical_cast<std::string>(1).c_str());

    // -- ML-specific options

    //PetscOptionsSetValue("-pc_ml_maxNlevels",
    //                      boost::lexical_cast<std::string>(4).c_str());

    //PetscOptionsSetValue("-pc_ml_CoarsenScheme", "METIS");


    //PetscOptionsSetValue("-pc_ml_maxCoarseSize",
    //                      boost::lexical_cast<std::string>(128).c_str());

    //PetscOptionsSetValue("-pc_ml_Threshold",
    //                      boost::lexical_cast<std::string>(2).c_str());

    //PetscOptionsSetValue("-pc_ml_PrintLevel",
    //                      boost::lexical_cast<std::string>(6).c_str());


    // Make sure options are set
    //PCSetFromOptions(pc);

    // Make sure the data structures have been constructed
    //std::cout << "!!! Set-up Create ksp" << std::endl;
    //KSPSetUp(*solver.ksp());
    //PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
    //std::cout << "!!! End view" << std::endl;

    //K
    //PCMGGetSmoother(PC pc,PetscInt l,KSP *ksp)

    //#if PETSC_VERSIOsyN_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    //PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    //PCFactorSetShiftAmount(pc, PETSC_DECIDE);
    //#else
    //PCFactorSetShiftNonzero(pc, PETSC_DECIDE);
    //#endif

    #else
    warning("PETSc has not been compiled with the ML library for   "
            "algerbraic multigrid. Default PETSc solver will be used. "
            "For performance, installation of ML is recommended.");
    #endif
  }
  else if (type == "additive_schwarz")
  {
    // Select method and overlap
    PCSetType(pc, _methods.find("additive_schwarz")->second);
    PCASMSetOverlap(pc, parameters("schwarz")["overlap"]);

    // Get sub-solvers and set sub-solver parameters
    KSP* sub_ksps;
    int num_local(0), first(0);
    PCASMGetSubKSP(pc, &num_local, &first, &sub_ksps);
    for (int i = 0; i < num_local; ++i)
    {
      // Get sub-preconditioner
      PC sub_pc;
      KSPGetPC(sub_ksps[i], &sub_pc);

      //PCSetType(sub_pc, PCLU);
      //PCFactorSetMatSolverPackage(sub_pc, MAT_SOLVER_UMFPACK);
      //PCSetType(sub_pc, PCILU);
      //KSPSetType(sub_ksps[i], KSPGMRES);
      PCFactorSetLevels(sub_pc, parameters("ilu")["fill_level"]);
      //PCFactorSetLevels(sub_pc, 4);
      //PCView(sub_pc, PETSC_VIEWER_STDOUT_WORLD);
    }
    //KSPSetTolerances(sub_ksps[0], 1.0e-1,
		//                      parameters["absolute_tolerance"],
		//                      parameters["divergence_limit"],
		//                      100);
    //KSPMonitorSet(sub_ksps[0], KSPMonitorTrueResidualNorm, 0, 0);
  }
  else if (type != "default")
  {
    PCSetType(pc, _methods.find(type)->second);
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    PCFactorSetShiftAmount(pc, parameters["shift_nonzero"]);
    #else
    PCFactorSetShiftNonzero(pc, parameters["shift_nonzero"]);
    #endif
  }

  PCFactorSetLevels(pc, parameters("ilu")["fill_level"]);

  // Make sure options are set
  PCSetFromOptions(pc);

  if (parameters["report"])
    PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
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
