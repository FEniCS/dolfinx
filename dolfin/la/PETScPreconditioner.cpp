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

#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "PETScKrylovSolver.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
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
const std::vector<std::pair<std::string, std::string> >
PETScPreconditioner::_methods_descr
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
  p_ml.add<std::size_t>("print_level", 0, 10);
  p_ml.add<std::string>("cycle_type", boost::assign::list_of("v")("w"));
  p_ml.add<std::size_t>("max_num_levels");
  p_ml.add<std::size_t>("max_coarse_size");
  p_ml.add<double>("aggregation_damping_factor");
  p_ml.add<double>("threshold");
  p_ml.add<bool>("use_spectral_Anorm");
  p_ml.add<int>("energy_minimization", -1, 4);
  p_ml.add<double>("energy_minimization_threshold");
  p_ml.add<double>("auxiliary_threshold");
  p_ml.add<bool>("repartition");
  p_ml.add<std::string>("repartition_type",
                        boost::assign::list_of("Zoltan")("ParMETIS"));
  p_ml.add<std::string>("zoltan_repartition_scheme",
               boost::assign::list_of("RCB")("hypergraph")("fast_hypergraph"));
  p_ml.add<std::string>("aggregation_scheme",
            boost::assign::list_of("Uncoupled")("Coupled")("MIS")("METIS"));
  p.add(p_ml);

  // PETSc GAMG parameters
  Parameters p_gamg("gamg");
  p_gamg.add<std::size_t>("verbose");
  p_gamg.add("num_aggregation_smooths", 1);
  p_gamg.add<double>("threshold");
  p_gamg.add<std::size_t>("max_coarse_size");
  p_gamg.add<std::size_t>("max_process_coarse_size");
  p_gamg.add<bool>("repartition");
  p_gamg.add<bool>("square_graph");
  p_gamg.add<bool>("symmetrize_graph");
  p_gamg.add<std::size_t>("max_num_levels");
  p.add(p_gamg);

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
  p_boomeramg.add<std::string>("relax_type_coarse");

  // Hypre package parameters
  Parameters p_hypre("hypre");
  p_hypre.add(p_parasails);
  p_hypre.add(p_boomeramg);
  p.add(p_hypre);

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
  ierr = KSPGetPC(*(solver.ksp()), &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Treat special cases  first
  if (_type.find("hypre") != std::string::npos)
  {
    #if PETSC_HAVE_HYPRE
    ierr = PCSetType(pc, PCHYPRE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    ierr = PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftType");
    ierr = PCFactorSetShiftAmount(pc, PETSC_DECIDE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftAmount");

    if (_type == "hypre_amg" || _type == "amg")
    {
      ierr = PCHYPRESetType(pc, "boomeramg");
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPRESetType");
      if (parameters("mg")["num_sweeps"].is_set())
      {
        const std::size_t num_sweeps = parameters("mg")["num_sweeps"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_grid_sweeps_all",
                         boost::lexical_cast<std::string>(num_sweeps).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["cycle_type"].is_set())
      {
        const std::string cycle
          = parameters("hypre")("BoomerAMG")["cycle_type"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_cycle_type", cycle.c_str());
      }
      if (parameters("hypre")("BoomerAMG")["max_levels"].is_set())
      {
        const std::size_t max_levels
          = parameters("hypre")("BoomerAMG")["max_levels"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_max_levels",
                          boost::lexical_cast<std::string>(max_levels).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["strong_threshold"].is_set())
      {
        const double threshold
          = parameters("hypre")("BoomerAMG")["strong_threshold"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold",
                          boost::lexical_cast<std::string>(threshold).c_str());
      }
      if (parameters("hypre")("BoomerAMG")["relaxation_weight"].is_set())
      {
        const double relax
          = parameters("hypre")("BoomerAMG")["relaxation_weight"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_relax_weight_all",
                            boost::lexical_cast<std::string>(relax).c_str() );
      }
      if (parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"].is_set())
      {
        const std::size_t levels
          = parameters("hypre")("BoomerAMG")["agressive_coarsening_levels"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_agg_nl",
                            boost::lexical_cast<std::string>(levels).c_str() );
      }
      if (parameters("hypre")("BoomerAMG")["relax_type_coarse"].is_set())
      {
        const std::string type
          = parameters("hypre")("BoomerAMG")["relax_type_coarse"];
        PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse",
                             type.c_str());
      }
    }
    else if (_type == "hypre_parasails")
    {
      ierr = PCHYPRESetType(pc, "parasails");
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPRESetType");
      if (parameters("hypre")("parasails")["threshold"].is_set())
      {
        const double thresh = parameters("hypre")("parasails")["threshold"];
        PetscOptionsSetValue("-pc_hypre_parasails_thresh",
                             boost::lexical_cast<std::string>(thresh).c_str());
      }
      if (parameters("hypre")("parasails")["levels"].is_set())
      {
        const int levels = parameters("hypre")("parasails")["levels"];
        PetscOptionsSetValue("-pc_hypre_parasails_nlevels",
                             boost::lexical_cast<std::string>(levels).c_str());
      }
    }
    else if (_type == "hypre_euclid")
    {
      ierr = PCHYPRESetType(pc, "euclid");
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPRESetType");
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
  else if (_type == "amg_ml" || _type == "ml_amg")
  {
    #if PETSC_HAVE_ML

    // Set preconditioner to ML
    ierr = PCSetType(pc, PCML);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

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

    // Spectral norm
    if (parameters("ml")["use_spectral_Anorm"].is_set())
    {
      const bool Anorm = parameters("ml")["use_spectral_Anorm"];
      if (Anorm)
        PetscOptionsSetValue("-pc_ml_SpectralNormScheme_Anorm", "1");
      else
        PetscOptionsSetValue("-pc_ml_SpectralNormScheme_Anorm", "0");
    }

    // Energy minimization strategy
    if (parameters("ml")["energy_minimization"].is_set())
    {
      const int method = parameters("ml")["energy_minimization"];
      PetscOptionsSetValue("-pc_ml_EnergyMinimization",
                            boost::lexical_cast<std::string>(method).c_str());

      // Energy minimization drop tolerance
      if (parameters("ml")["energy_minimization_threshold"].is_set())
      {
        const double threshold
          = parameters("ml")["energy_minimization_threshold"];
        PetscOptionsSetValue("-pc_ml_EnergyMinimizationDropTol",
                          boost::lexical_cast<std::string>(threshold).c_str());
      }
    }

    // Auxiliary threshold drop tolerance
    /*
    PetscOptionsSetValue("-pc_ml_Aux", "1");
    if (parameters("ml")["auxiliary_threshold"].is_set())
    {
      const double threshold = parameters("ml")["auxiliary_threshold"];
      PetscOptionsSetValue("-pc_ml_AuxThreshold",
                            boost::lexical_cast<std::string>(threshold).c_str());
    }
    */

    // Allow ML to re-partition problem
    if (parameters("ml")["repartition"].is_set())
    {
      const bool repartition = parameters("ml")["repartition"];
      if (repartition)
      {
        PetscOptionsSetValue("-pc_ml_repartition", "1");
        if (parameters("ml")["repartition_type"].is_set())
        {
          PetscOptionsSetValue("-pc_ml_repartitionType",
                   parameters("ml")["repartition_type"].value_str().c_str());
        }
        if (parameters("ml")["zoltan_repartition_scheme"].is_set())
        {
          PetscOptionsSetValue("-pc_ml_repartitionZoltanScheme",
           parameters("ml")["zoltan_repartition_scheme"].value_str().c_str());
        }
      }
      else
        PetscOptionsSetValue("-pc_ml_repartition", "0");
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
                          boost::lexical_cast<std::string>(4).c_str());

    //PetscOptionsSetValue("-mg_levels_pc_type", "none");
    PetscOptionsSetValue("-mg_levels_pc_type", "jacobi");
    //PetscOptionsSetValue("-mg_levels_pc_type", "sor");
    //PetscOptionsSetValue("-mg_levels_1_pc_sor_its",
    //                      boost::lexical_cast<std::string>(2).c_str());

    /*
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
    */

    #else
    warning("PETSc has not been compiled with the ML library for   "
            "algerbraic multigrid. Default PETSc solver will be used. "
            "For performance, installation of ML is recommended.");
    #endif
  }
  else if (_type == "petsc_amg")
  {
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 2

    // The PETSc AMG (smoothed aggegration) preconditioner
    //PetscOptionsSetValue("-log_summary",
    //                     boost::lexical_cast<std::string>(1).c_str());

    // Set preconditioner to ML
    ierr = PCSetType(pc, PCGAMG);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    // Coarse level solver
    #if PETSC_HAVE_MUMPS
    PetscOptionsSetValue("-mg_coarse_ksp_type", "preonly");
    PetscOptionsSetValue("-mg_coarse_pc_type", "lu");
    PetscOptionsSetValue("-mg_coarse_pc_factor_mat_solver_package", "mumps");
    #endif

    // Output level
    if (parameters("gamg")["verbose"].is_set())
    {
      const std::size_t verbose = parameters("gamg")["verbose"];
      PetscOptionsSetValue("-pc_gamg_verbose",
                  boost::lexical_cast<std::string>(verbose ?  1 : 0 ).c_str());
    }

    // -------- Aggregation options

    // Set to smoothed aggregation
    PetscOptionsSetValue("-pc_gamg_type", "agg");

    // Number of aggregation smooths
    if (parameters("gamg")["num_aggregation_smooths"].is_set())
    {
      const std::size_t num_smooths
        = parameters("gamg")["num_aggregation_smooths"];
      PetscOptionsSetValue("-pc_gamg_agg_nsmooths",
                        boost::lexical_cast<std::string>(num_smooths).c_str());
      // Note: Below seems to have no effect
      //PCGAMGSetNSmooths(pc, num_smooths);
    }

    // Square graph
    if (parameters("gamg")["square_graph"].is_set())
    {
      const bool square = parameters("gamg")["square_graph"];
      PetscOptionsSetValue("-pc_gamg_square_graph",
                         boost::lexical_cast<std::string>(square).c_str());
      //PCGAMGSetSquareGraph(pc, square ? PETSC_TRUE : PETSC_FALSE);
    }

    // Symmetrize graph (if not square)
    if (parameters("gamg")["symmetrize_graph"].is_set())
    {
      const bool symmetrize = parameters("gamg")["symmetrize_graph"];
      PetscOptionsSetValue("-pc_gamg_sym_graph",
                           boost::lexical_cast<std::string>(symmetrize).c_str());
      //PCGAMGSetSymGraph(pc, symmetric ? PETSC_TRUE : PETSC_FALSE);
    }

    // -------- AMG options

    PetscOptionsSetValue("-mg_levels_ksp_max_it",
                          boost::lexical_cast<std::string>(4).c_str());
    PetscOptionsSetValue("-mg_levels_pc_type", "jacobi");

    // Threshold parameters used in aggregation
    if (parameters("gamg")["threshold"].is_set())
    {
      const double threshold = parameters("gamg")["threshold"];
      ierr = PCGAMGSetThreshold(pc, threshold);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCGAMGSetThreshold");
    }

    // Maximum coarse level problem size
    if (parameters("gamg")["max_coarse_size"].is_set())
    {
      const std::size_t max_size = parameters("gamg")["max_coarse_size"];
      ierr = PCGAMGSetCoarseEqLim(pc, max_size);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCGAMGSetCoarseEqLimit");
    }

    // Maximum coarse level problem size on a process
    if (parameters("gamg")["max_process_coarse_size"].is_set())
    {
      const std::size_t max_size
        = parameters("gamg")["max_process_coarse_size"];
      ierr = PCGAMGSetProcEqLim(pc, max_size);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCGAMGSetProcEqLim");
    }

    // Allow GAMG to re-partition problem
    if (parameters("gamg")["repartition"].is_set())
    {
      const bool repartition = parameters("gamg")["repartition"];
      ierr = PCGAMGSetRepartitioning(pc,repartition ? PETSC_TRUE : PETSC_FALSE);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCGAMGSetRepartitioning");
    }

    // Maximum numebr of levels
    if (parameters("gamg")["max_num_levels"].is_set())
    {
      const std::size_t num_levels = parameters("gamg")["max_num_levels"];
      PetscOptionsSetValue("-pc_mg_levels",
                         boost::lexical_cast<std::string>(num_levels).c_str());
      // Note: Below doesn't appear to work
      //ierr = PCGAMGSetNlevels(pc, num_levels);
      //if (ierr != 0) petsc_error(ierr, __FILE__, "PCGAMGSetNlevels");
    }

    #else
    warning("PETSc native algebraic multigrid support requires PETSc"
             "version > 3.2. Default PETSc preconditioner will be used.");
    #endif

  }
  else if (_type == "additive_schwarz")
  {
    // Select method and overlap
    ierr = PCSetType(pc, _methods.find("additive_schwarz")->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");

    const int schwarz_overlap = parameters("schwarz")["overlap"];
    PCASMSetOverlap(pc, schwarz_overlap);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCASMSetOverlap");

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

  // Make sure options are set
  ierr = PCSetFromOptions(pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetFromOptions");

  // Set physical coordinates for row dofs
  if (!_coordinates.empty())
  {
    dolfin_assert(gdim > 0);
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3
    ierr = PCSetCoordinates(pc, gdim, _coordinates.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetCoordinates");
    #else
    ierr = PCSetCoordinates(pc, gdim, _coordinates.size()/gdim,
                            _coordinates.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetCoordinates");
    #endif
  }

  // Clear memory
  _coordinates.clear();

  // Print preconditioner information
  const bool report = parameters["report"];
  if (report)
  {
    ierr = PCSetUp(pc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetUp");
    ierr = PCView(pc, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCView");
  }
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_nullspace(const VectorSpaceBasis& near_nullspace)
{
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3
  dolfin_error("PETScPreconditioner.cpp",
               "set approximate null space for PETSc preconditioner",
               "This is supported by PETSc version > 3.2");
  #else

  // Clear near nullspace
  petsc_near_nullspace.reset();
  _near_nullspace.clear();

  // Copy vectors
  for (std::size_t i = 0; i < near_nullspace.dim(); ++i)
  {
    dolfin_assert(near_nullspace[i]);
    const PETScVector& x = near_nullspace[i]->down_cast<PETScVector>();

    // Copy vector
    _near_nullspace.push_back(x);
  }

  // Get pointers to underlying PETSc objects
  std::vector<Vec> petsc_vec(near_nullspace.dim());
  for (std::size_t i = 0; i < near_nullspace.dim(); ++i)
    petsc_vec[i] = _near_nullspace[i].vec();

  // Create null space
  petsc_near_nullspace.reset(new MatNullSpace, PETScMatNullSpaceDeleter());
  PetscErrorCode ierr;
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, near_nullspace.dim(),
                            petsc_vec.data(), petsc_near_nullspace.get());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatNullSpaceCreate");
  #endif
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_coordinates(const std::vector<double>& x,
                                          std::size_t dim)
{
  _coordinates = x;
  gdim = dim;
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::set_fieldsplit(PETScKrylovSolver& solver,
                    const std::vector<std::vector<dolfin::la_index> >& fields,
                    const std::vector<std::string>& split_names)
{
  dolfin_assert(fields.size() == split_names.size());
  PetscErrorCode ierr;

  if (fields.empty())
    return;

  // Get PETSc PC pointer
  PC pc;
  dolfin_assert(solver.ksp());
  ierr = KSPGetPC(*(solver.ksp()), &pc);
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
    PCFieldSplitSetIS(pc, split_names[i].c_str(), is);
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
