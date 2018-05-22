// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
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

/// Initialise matrix. Matrix is not zeroed.
void init(la::PETScMatrix& A, const Form& a);

/// Initialise nested (MatNest) matrix. Matrix is not zeroed.
void init_nest(la::PETScMatrix& A,
               std::vector<std::vector<const fem::Form*>> a);

/// Initialise nested (VecNest) vector. Vector is not zeroed.
la::PETScVector init_nest(std::vector<const fem::Form*> L);

/// Initialise monolithic  matrix. Matrix is not zeroed.
void init_monolithic(la::PETScMatrix& A,
                     std::vector<std::vector<const fem::Form*>> a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector init_monolithic(std::vector<const fem::Form*> L);

/// Initialise vector. Vector is not zeroed.
la::PETScVector init_vector(const Form& L);

/// Get new global index in 'spliced' indices
std::size_t get_global_index(const std::vector<const common::IndexMap*> maps,
                             const unsigned int field, const unsigned int n);
} // namespace fem
} // namespace dolfin
