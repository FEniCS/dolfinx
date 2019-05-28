// Copyright (C) 2008-2009 Igor A. Baratta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Graph.h"
#include <cstddef>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <map>
#include <utility>

namespace dolfin
{

namespace mesh
{
class CellType;
}

namespace graph
{

template <typename T>
class CSRGraph;

/// This class provides an interface to KaHIP

class KaHIP
{
#ifdef HAS_KAHIP
public:
  // Standard KaHIP partition
  void partition(MPI_Comm mpi_comm,
                 const CSRGraph<unsigned long long>& csr_graph);

#endif
};
} // namespace graph
} // namespace dolfin
