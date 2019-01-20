// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <memory>
#include <vector>

namespace dolfin
{
namespace common
{
class IndexMap;
}

namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la
namespace function
{
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
class MeshGeometry;
} // namespace mesh

namespace fem
{
class Form;

/// Compute IndexMaps for stacked index maps
std::vector<std::vector<std::shared_ptr<const common::IndexMap>>>
blocked_index_sets(const std::vector<std::vector<const fem::Form*>> a);

/// Initialise matrix. Matrix is not zeroed.
la::PETScMatrix init_matrix(const Form& a);

/// Initialise nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix init_nest_matrix(std::vector<std::vector<const fem::Form*>> a);

/// Initialise nested (VecNest) vector. Vector is not zeroed.
la::PETScVector init_nest(std::vector<const fem::Form*> L);

/// Initialise monolithic  matrix. Matrix is not zeroed.
la::PETScMatrix
init_monolithic_matrix(std::vector<std::vector<const fem::Form*>> a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector init_monolithic(std::vector<const fem::Form*> L);

/// Get new global index in 'spliced' indices
std::size_t get_global_index(const std::vector<const common::IndexMap*> maps,
                             const unsigned int field, const unsigned int n);
} // namespace fem
} // namespace dolfin
