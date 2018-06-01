// Copyright (C) 2009-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Parameters.h"

namespace dolfin
{
namespace parameter
{

/// This class defines the global DOLFIN parameter database.

class GlobalParameters : public Parameters
{
public:
  /// Constructor
  GlobalParameters();

  /// Destructor
  virtual ~GlobalParameters();

  /// Parse parameters from command-line
  virtual void parse(int argc, char* argv[]);

  /// Default parameter values
  static Parameters default_parameters()
  {
    Parameters p("dolfin");

    //-- Output

    // Print standard output on all processes
    p.add("std_out_all_processes", true);

    // Add dof ordering library
    std::string default_dof_ordering_library = "Boost";
#ifdef HAS_SCOTCH
    default_dof_ordering_library = "SCOTCH";
#endif
    p.add("dof_ordering_library", default_dof_ordering_library,
          {"Boost", "random", "SCOTCH"});

    //-- Meshes

    // Mesh ordering via SCOTCH and GPS
    p.add("reorder_cells_gps", false);
    p.add("reorder_vertices_gps", false);

    // Set default graph/mesh partitioner
    std::string default_mesh_partitioner = "SCOTCH";
#ifdef HAS_PARMETIS
#ifndef HAS_SCOTCH
    default_mesh_partitioner = "ParMETIS";
#endif
#endif
    p.add("mesh_partitioner", default_mesh_partitioner,
          {"ParMETIS", "SCOTCH", "None"});

    // Approaches to partitioning (following Zoltan syntax)
    // but applies to ParMETIS
    p.add("partitioning_approach", "PARTITION",
          {"PARTITION", "REPARTITION", "REFINE"});

    return p;
  }
};

/// The global parameter database
extern GlobalParameters parameters;
} // namespace parameter
} // namespace dolfin
