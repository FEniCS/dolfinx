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
#include <string>
#include <utility>
#include <vector>

#ifndef PARHIP_INTERFACE
#define PARHIP_INTERFACE
#endif

extern "C"
{
#include "KaHIP_interface.h"
}

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
#ifdef PARHIP_INTERFACE
public:
  // Standard ParMETIS partition
  static std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
  partition(MPI_Comm mpi_comm, const CSRGraph<idx_t>& csr_graph);

#endif
};
} // namespace graph
} // namespace dolfin
