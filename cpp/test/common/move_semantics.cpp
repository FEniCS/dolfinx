// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Library types are moved routinely, and are stored in standard
// containers that fall back to copying on reallocation unless the move
// is non-throwing. The move operations are all `= default`, so this
// property is implicit: it holds only as long as every member's move is
// itself non-throwing. Adding a member with a throwing move would
// silently degrade moves to copies, so the property is asserted here
// rather than left to chance.

#include <basix/finite-element.h>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/Table.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <type_traits>

using namespace dolfinx;

namespace
{
/// @brief Non-throwing move construction, required of every type.
template <typename T>
constexpr bool nothrow_move_c = std::is_nothrow_move_constructible_v<T>;

/// @brief Non-throwing move assignment. Not asserted for io::XDMFFile,
/// which does not provide a move assignment operator, nor for
/// fem::Expression, whose std::function member is not required by the
/// standard to be non-throwing on move assignment (libc++ and libstdc++
/// both make it so, but that is not guaranteed).
template <typename T>
constexpr bool nothrow_move_a = std::is_nothrow_move_assignable_v<T>;

// Classes templated on a scalar type
template <dolfinx::scalar T>
struct scalar_classes
{
  static_assert(nothrow_move_c<fem::Constant<T>>);
  static_assert(nothrow_move_c<fem::DirichletBC<T>>);
  static_assert(nothrow_move_c<fem::Expression<T>>);
  static_assert(nothrow_move_c<fem::Form<T>>);
  static_assert(nothrow_move_c<fem::Function<T>>);
  static_assert(nothrow_move_c<la::MatrixCSR<T>>);
  static_assert(!std::is_copy_constructible_v<la::MatrixCSR<T>>);
  static_assert(!std::is_copy_assignable_v<la::MatrixCSR<T>>);
  static_assert(nothrow_move_c<la::Vector<T>>);

  static_assert(nothrow_move_a<fem::Constant<T>>);
  static_assert(nothrow_move_a<fem::DirichletBC<T>>);
  static_assert(nothrow_move_a<fem::Form<T>>);
  static_assert(std::is_move_assignable_v<fem::Expression<T>>);
  static_assert(nothrow_move_a<fem::Function<T>>);
  static_assert(nothrow_move_a<la::MatrixCSR<T>>);
  static_assert(nothrow_move_a<la::Vector<T>>);
};

template struct scalar_classes<float>;
template struct scalar_classes<double>;
template struct scalar_classes<std::complex<float>>;
template struct scalar_classes<std::complex<double>>;

// Classes templated on a geometry (floating point) type
template <std::floating_point T>
struct geometry_classes
{
  static_assert(nothrow_move_c<fem::CoordinateElement<T>>);
  static_assert(nothrow_move_c<fem::FiniteElement<T>>);
  static_assert(nothrow_move_c<fem::FunctionSpace<T>>);
  static_assert(nothrow_move_c<geometry::BoundingBoxTree<T>>);
  static_assert(nothrow_move_c<mesh::Geometry<T>>);
  static_assert(nothrow_move_c<mesh::Mesh<T>>);

  static_assert(nothrow_move_a<fem::CoordinateElement<T>>);
  static_assert(nothrow_move_a<fem::FiniteElement<T>>);
  static_assert(nothrow_move_a<fem::FunctionSpace<T>>);
  static_assert(nothrow_move_a<geometry::BoundingBoxTree<T>>);
  static_assert(nothrow_move_a<mesh::Geometry<T>>);
  static_assert(nothrow_move_a<mesh::Mesh<T>>);
};

template struct geometry_classes<float>;
template struct geometry_classes<double>;

// Classes templated on a mesh tag value type
template <typename T>
struct meshtags_classes
{
  static_assert(nothrow_move_c<mesh::MeshTags<T>>);
  static_assert(nothrow_move_a<mesh::MeshTags<T>>);
};

template struct meshtags_classes<std::int8_t>;
template struct meshtags_classes<std::int32_t>;
template struct meshtags_classes<std::int64_t>;
template struct meshtags_classes<double>;

// Non-templated classes, and templates over a fixed index type
static_assert(nothrow_move_c<common::IndexMap>);
static_assert(nothrow_move_c<common::Scatterer<>>);
static_assert(nothrow_move_c<dolfinx::MPI::Comm>);
static_assert(nothrow_move_c<dolfinx::MPI::Datatype<double>>);
static_assert(nothrow_move_c<dolfinx::Table>);
static_assert(nothrow_move_c<common::Timer<>>);
static_assert(nothrow_move_c<fem::DofMap>);
static_assert(nothrow_move_c<fem::ElementDofLayout>);
static_assert(nothrow_move_c<graph::AdjacencyList<std::int32_t>>);
static_assert(nothrow_move_c<graph::AdjacencyList<std::int64_t>>);
static_assert(nothrow_move_c<io::VTKFile>);
static_assert(nothrow_move_c<io::XDMFFile>);
static_assert(nothrow_move_c<la::SparsityPattern>);
static_assert(nothrow_move_c<mesh::EntityMap>);
static_assert(nothrow_move_c<mesh::Topology>);

static_assert(nothrow_move_a<common::IndexMap>);
static_assert(nothrow_move_a<common::Scatterer<>>);
static_assert(nothrow_move_a<dolfinx::MPI::Comm>);
static_assert(nothrow_move_a<dolfinx::MPI::Datatype<double>>);
static_assert(nothrow_move_a<dolfinx::Table>);
static_assert(nothrow_move_a<common::Timer<>>);
static_assert(nothrow_move_a<fem::DofMap>);
static_assert(nothrow_move_a<fem::ElementDofLayout>);
static_assert(nothrow_move_a<graph::AdjacencyList<std::int32_t>>);
static_assert(nothrow_move_a<graph::AdjacencyList<std::int64_t>>);
static_assert(nothrow_move_a<io::VTKFile>);
static_assert(nothrow_move_a<la::SparsityPattern>);
static_assert(nothrow_move_a<mesh::EntityMap>);
static_assert(nothrow_move_a<mesh::Topology>);
} // namespace
