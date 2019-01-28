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

/// Create matrix. Matrix is not zeroed.
la::PETScMatrix create_matrix(const Form& a);

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
la::PETScMatrix
create_matrix_block(std::vector<std::vector<const fem::Form*>> a);

/// Create nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix
create_matrix_nest(std::vector<std::vector<const fem::Form*>> a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector create_vector_block(std::vector<const fem::Form*> L);

/// Initialise nested (VecNest) vector. Vector is not zeroed.
la::PETScVector create_vector_nest(std::vector<const fem::Form*> L);

/// Get new global index in 'spliced' indices
std::size_t get_global_index(const std::vector<const common::IndexMap*> maps,
                             const unsigned int field, const unsigned int n);
} // namespace fem
} // namespace dolfin
