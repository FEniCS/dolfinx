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
#include <dolfin/log/dolfin_log.h>
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "PETScKrylovSolver.h"
#include "PETScVector.h"
#include "PETScPreconditioner.h"

using namespace dolfin;

class PETScMatNullSpaceDeleter
{
public:
  void operator() (MatNullSpace* ns)
  {
    if (*ns)
      MatNullSpaceDestroy(ns);
    delete ns;
  }
};

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
                              #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 2
                              ("petsc_amg",        PCGAMG)
                              #endif  
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
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 2
    ("petsc_amg",        "PETSc algebraic multigrid")
    #endif
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

  // General parameters
  //p.add("view", false);

  // Generic multigrid parameters
  Parameters p_mg("mg");
  p_mg.add<std::size_t>("num_levels");
  p_mg.add<std::size_t>("num_sweeps");
  p.add(p_mg);

  // ML package parameters
  Parameters p_ml("ml");
  p_ml.add<std::size_t>("max_coarse_size");
  p_ml.add<double>("aggregation_damping_factor");
  p_ml.add<double>("threshold");
  p_ml.add<std::size_t>("max_num_levels");
  p_ml.add<std::size_t>("print_level", 0, 10);

  std::set<std::string> ml_schemes;
  ml_schemes.insert("v");
  ml_schemes.insert("w");
  p_ml.add<std::string>("cycle_type", ml_schemes);

  std::set<std::string> aggregation_schemes;
  aggregation_schemes.insert("Uncoupled");
  aggregation_schemes.insert("Coupled");
  aggregation_schemes.insert("MIS");
  aggregation_schemes.insert("METIS");
  p_ml.add<std::string>("aggregation_scheme", aggregation_schemes);
  p.add(p_ml);

  // Hypre/ParaSails parameters
  Parameters p_parasails("parasails");
  p_parasails.add<double>("threshold");
  p_parasails.add<std::size_t>("levels");

  // Hypre/BoomerAMG parameters
  Parameters p_boomeramg("BoomerAMG");
  p_boomeramg.add<std::string>("cycle_type"); // "V" or "W"
  p_boomeramg.add<std::size_t>("max_levels");
  p_boomeramg.add<double>("strong_threshold");
  p_boomeramg.add<double>("relaxation_weight");
  p_boomeramg.add<std::size_t>("agressive_coarsening_levels");

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

    PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    PCFactorSetShiftAmount(pc, PETSC_DECIDE);

    if (type == "hypre_amg" || type == "amg")
    {
      PCHYPRESetType(pc, "boomeramg");
      if (parameters("mg")["num_sweeps"].is_set())
      {
        const std::size_t num_sweeps = parameters("mg")["num_sweeps"];
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
        const std::size_t max_levels = parameters("hypre")("BoomerAMG")["max_levels"];
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
        const double relax = parameters("hypre")("BoomerAMG")["relaxation_weight"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_relax_weight_all",
                            boost::lexical_cast<std::string>(relax).c_str() );
      }
      if (parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"].is_set())
      {
        const std::size_t levels = parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"];
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
      const std::size_t ilu_level = parameters("ilu")["fill_level"];
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

    // Set preconditioner to ML
    PCSetType(pc, PCML);

    // Set some parameters before set-up. These parameters can:
    // (i)  Only be set via the PETSc parameters system; or
    // (ii) Cannot be changed once once the preconditioner has been
    //      setup. The is tricky because we don't know how many levels ML
    //      will create until we've called PCSetUp

    // --- ML parameters

    // Output level
    if (parameters("ml")["print_level"].is_set())
    {
      const std::size_t print_level = parameters("ml")["print_level"];
      PetscOptionsSetValue("-pc_ml_PrintLevel",
                           boost::lexical_cast<std::string>(print_level).c_str());
    }

    // Maximum number of levels
    if (parameters("ml")["max_num_levels"].is_set())
    {
      const std::size_t max_levels = parameters("ml")["max_num_levels"];
      PetscOptionsSetValue("-pc_ml_maxNlevels",
                           boost::lexical_cast<std::string>(max_levels).c_str());
    }

    // Aggregation scheme (Uncoupled, Coupled, MIS, . . .)
    if (parameters("ml")["aggregation_scheme"].is_set())
    {
      const std::string scheme = parameters("ml")["aggregation_scheme"];
      PetscOptionsSetValue("-pc_ml_CoarsenScheme", scheme.c_str());
    }

    // Aggregation damping factor
    if (parameters("ml")["aggregation_damping_factor"].is_set())
    {
      const double damping = parameters("ml")["aggregation_damping_factor"];
      PetscOptionsSetValue("-pc_ml_DampingFactor",
                           boost::lexical_cast<std::string>(damping).c_str());
    }

    // Maximum coarse level problem size
    if (parameters("ml")["max_coarse_size"].is_set())
    {
      const std::size_t max_size = parameters("ml")["max_coarse_size"];
      PetscOptionsSetValue("-pc_ml_maxCoarseSize",
                            boost::lexical_cast<std::string>(max_size).c_str());
    }

    // Threshold parameters used in aggregation
    if (parameters("ml")["threshold"].is_set())
    {
      const double threshold = parameters("ml")["threshold"];
      PetscOptionsSetValue("-pc_ml_Threshold",
                            boost::lexical_cast<std::string>(threshold).c_str());
    }

    // --- PETSc parameters

    // Number of smmoother applications
    if (parameters("mg")["num_sweeps"].is_set())
    {
      const std::size_t num_sweeps = parameters("mg")["num_sweeps"];
      PetscOptionsSetValue("-pc_mg_smoothup",
                           boost::lexical_cast<std::string>(num_sweeps).c_str());
      PetscOptionsSetValue("-pc_mg_smoothdown",
                           boost::lexical_cast<std::string>(num_sweeps).c_str());
    }

    if (parameters("ml")["cycle_type"].is_set())
    {
      const std::string type = parameters("mg")["cycle_type"];
      PetscOptionsSetValue("-pc_mg_cycles", type.c_str());
    }

    // Coarse level solver
    #if PETSC_HAVE_MUMPS
    PetscOptionsSetValue("-mg_coarse_ksp_type", "preonly");
    PetscOptionsSetValue("-mg_coarse_pc_type", "lu");
    PetscOptionsSetValue("-mg_coarse_pc_factor_mat_solver_package", "mumps");
    #endif

    // Smoother on all levels
    PetscOptionsSetValue("-mg_levels_ksp_type", "chebyshev");
    //PetscOptionsSetValue("mg_levels_ksp_chebyshev_estimate_eigenvalues",
    //                      "0.0,1.1");
    //PetscOptionsSetValue("-mg_levels_ksp_type", "richardson");
    PetscOptionsSetValue("-mg_levels_ksp_max_it",
                          boost::lexical_cast<std::string>(3).c_str());

    //PetscOptionsSetValue("-mg_levels_pc_type", "none");
    PetscOptionsSetValue("-mg_levels_pc_type", "jacobi");
    //PetscOptionsSetValue("-mg_levels_pc_type", "sor");
    //PetscOptionsSetValue("-mg_levels_1_pc_sor_its",
    //                      boost::lexical_cast<std::string>(2).c_str());

    // Level 1 smoother (0 is coarse, N is finest)
    PetscOptionsSetValue("-mg_levels_1_ksp_type", "chebyshev");
    PetscOptionsSetValue("-mg_levels_1_ksp_max_it",
                          boost::lexical_cast<std::string>(1).c_str());
    PetscOptionsSetValue("-mg_levels_1_pc_type", "sor");
    PetscOptionsSetValue("-mg_levels_1_pc_sor_its",
                          boost::lexical_cast<std::string>(2).c_str());

    // Level 2  smoother
    PetscOptionsSetValue("-mg_levels_2_ksp_type", "chebyshev");
    PetscOptionsSetValue("-mg_levels_2_ksp_max_it",
                          boost::lexical_cast<std::string>(1).c_str());
    PetscOptionsSetValue("-mg_levels_2_pc_type", "sor");
    PetscOptionsSetValue("-mg_levels_2_pc_sor_its",
			 boost::lexical_cast<std::string>(1).c_str());

    #else
    warning("PETSc has not been compiled with the ML library for   "
            "algerbraic multigrid. Default PETSc solver will be used. "
            "For performance, installation of ML is recommended.");
    #endif
  }
  else if (type == "petsc_amg")
  {
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 2

    // The PETSc AMG (smoothed aggegration) preconditioner
    PetscOptionsSetValue("-log_summary",
                         boost::lexical_cast<std::string>(1).c_str());

    // Set preconditioner to ML
    PCSetType(pc, PCGAMG);

    // Coarse level solver
    #if PETSC_HAVE_MUMPS
    PetscOptionsSetValue("-mg_coarse_ksp_type", "preonly");
    PetscOptionsSetValue("-mg_coarse_pc_type", "lu");
    PetscOptionsSetValue("-mg_coarse_pc_factor_mat_solver_package", "mumps");
    #endif

    /*
    if (parameters("mg")["num_levels"].is_set())
    {
      const uint num_levels = parameters("mg")["num_levels"];
      PetscOptionsSetValue("-pc_mg_num_levels",
                           boost::lexical_cast<std::string>(num_levels).c_str());
    }
    */

    // Set to smoothed aggregation
    PetscOptionsSetValue("-pc_gamg_type", "agg");

    // Number of smoother applications
    PetscOptionsSetValue("-pc_gamg_agg_nsmooths",
                         boost::lexical_cast<std::string>(1).c_str());

    //PetscOptionsSetValue("mg_levels_ksp_chebyshev_estimate_eigenvalues",
    //                      "0.1,1.1");

    //PetscOptionsSetValue("-pc_gamg_verbose",
    //                     boost::lexical_cast<std::string>(2).c_str());

    PetscOptionsSetValue("-mg_levels_ksp_max_it",
                          boost::lexical_cast<std::string>(2).c_str());
    PetscOptionsSetValue("-mg_levels_pc_type", "jacobi");

    PetscOptionsSetValue("-pc_gamg_threshold",
                         boost::lexical_cast<std::string>(0.01).c_str());

    //PetscOptionsSetValue("-pc_gamg_eigtarget",
    //                      "0.1,1.1");

    PetscOptionsSetValue("-pc_gamg_coarse_eq_limit",
                         boost::lexical_cast<std::string>(2048).c_str());

    //PetscOptionsSetValue("-pc_gamg_process_eq_limit",
    //                     boost::lexical_cast<std::string>(16).c_str());

    //PetscOptionsSetValue("-pc_gamg_use_agg_gasm",
    //                     boost::lexical_cast<std::string>(1).c_str());

    //PetscOptionsSetValue("-pc_gamg_repartition",
    //                     boost::lexical_cast<std::string>(1).c_str());

    //PetscOptionsSetValue("-pc_gamg_sym_graph",
    //                     boost::lexical_cast<std::string>(1).c_str());

    PetscOptionsSetValue("-pc_gamg_square_graph",
                         boost::lexical_cast<std::string>(1).c_str());

    //PetscOptionsSetValue("-pc_mg_levels",
    //                     boost::lexical_cast<std::string>(4).c_str());

    //PCMGSetLevels(pc, 5, &PETSC_COMM_WORLD);
    //PCGAMGSetNlevels(pc, 5);
    //PCGAMGSetProcEqLim(pc, 1000);
    //PCGAMGSetSymGraph(pc, PETSC_TRUE);

    #else
    warning("PETSc native algebraic multigrid support requires PETSc"
             "version > 3.2. Default PETSc preconditioner will be used.");
    #endif

  }
  else if (type == "additive_schwarz")
  {
    // Select method and overlap
    PCSetType(pc, _methods.find("additive_schwarz")->second);
    const int schwarz_overlap = parameters("schwarz")["overlap"];
    PCASMSetOverlap(pc, schwarz_overlap);

    // Get sub-solvers and set sub-solver parameters
    /*
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
    */
  }
  else if (type != "default")
  {
    PCSetType(pc, _methods.find(type)->second);
    PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    PCFactorSetShiftAmount(pc, parameters["shift_nonzero"]);
  }

  const int ilu_levels = parameters("ilu")["fill_level"];
  PCFactorSetLevels(pc, ilu_levels);

  // Make sure options are set
  PCSetFromOptions(pc);

  // Print preconditioner information
  const bool report = parameters["report"];
  if (report)
  {
    PCSetUp(pc);
    PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
  }
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_nullspace(const std::vector<const GenericVector*> nullspace)
{
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3
  dolfin_error("PETScMatrix.cpp",
               "set approximate null space for PETSc matrix",
               "This is supported by PETSc version > 3.2");
  #else
  if (nullspace.empty())
  {
    // Clear nullspace
    petsc_nullspace.reset();
    _nullspace.clear();
  }
  else
  {
    // Copy vectors
    for (std::size_t i = 0; i < nullspace.size(); ++i)
    {
      dolfin_assert(nullspace[i]);
      const PETScVector& x = nullspace[i]->down_cast<PETScVector>();

      // Copy vector
      _nullspace.push_back(x);
    }

    // Get pointers to underlying PETSc objects
    std::vector<Vec> petsc_vec(nullspace.size());
    for (std::size_t i = 0; i < nullspace.size(); ++i)
      petsc_vec[i] = *(_nullspace[i].vec().get());

    // Create null space
    petsc_nullspace.reset(new MatNullSpace, PETScMatNullSpaceDeleter());
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, nullspace.size(),
                       &petsc_vec[0], petsc_nullspace.get());
  }
  #endif
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
