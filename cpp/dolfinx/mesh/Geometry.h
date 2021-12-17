// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <memory>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;

/// Geometry stores the geometry imposed on a mesh.

class Geometry
{
public:
  /// Constructor
  template <typename AdjacencyList32, typename Array, typename Vector64>
  Geometry(const std::shared_ptr<const common::IndexMap>& index_map,
           AdjacencyList32&& dofmap, const fem::CoordinateElement& element,
           Array&& x, int dim, Vector64&& input_global_indices)
      : _dim(dim), _dofmap(std::forward<AdjacencyList32>(dofmap)),
        _index_map(index_map), _cmap(element), _x(std::forward<Array>(x)),
        _input_global_indices(std::forward<Vector64>(input_global_indices))
  {
    // assert(_x.shape(1) > 0 and _x.shape(1) <= 3);
    assert(_x.size() % 3 == 0);
    if (_x.size() / 3 != _input_global_indices.size())
      throw std::runtime_error("Size mis-match");

    // // Make all geometry 3D
    // if (_dim != 3)
    // {
    //   xt::xtensor<double, 2> c
    //       = xt::zeros<double>({_x.shape(0), static_cast<std::size_t>(3)});

    //   // The below should work, but misbehaves with the Intel icpx compiler
    //   // xt::view(c, xt::all(), xt::range(0, _dim)) = _x;
    //   auto x_view = xt::view(c, xt::all(), xt::range(0, _dim));
    //   x_view.assign(_x);

    //   std::swap(c, _x);
    // }
  }

  /// Copy constructor
  Geometry(const Geometry&) = default;

  /// Move constructor
  Geometry(Geometry&&) = default;

  /// Destructor
  ~Geometry() = default;

  /// Copy Assignment
  Geometry& operator=(const Geometry&) = delete;

  /// Move Assignment
  Geometry& operator=(Geometry&&) = default;

  /// Return Euclidean dimension of coordinate system
  int dim() const;

  /// DOF map
  const graph::AdjacencyList<std::int32_t>& dofmap() const;

  /// Index map
  std::shared_ptr<const common::IndexMap> index_map() const;

  /// Geometry degrees-of-freedom
  // auto x() { return xt::adapt(_x, {_x.size() / 3, std::size_t(3)}); }

  /// Geometry degrees-of-freedom
  auto xt() const { return xt::adapt(_x, {_x.size() / 3, std::size_t(3)}); }

  /// Geometry degrees-of-freedom (new)
  const std::vector<double>& xnew() const { return _x; }

  /// The element that describes the geometry map
  /// @return The coordinate/geometry element
  const fem::CoordinateElement& cmap() const;

  /// Global user indices
  const std::vector<std::int64_t>& input_global_indices() const;

private:
  // Geometric dimension
  int _dim;

  // Map per cell for extracting coordinate data
  graph::AdjacencyList<std::int32_t> _dofmap;

  // IndexMap for geometry 'dofmap'
  std::shared_ptr<const common::IndexMap> _index_map;

  // The coordinate element
  fem::CoordinateElement _cmap;

  // Coordinates for all points stored as a contiguous array
  // xt::xtensor<double, 2> _x;
  std::vector<double> _x;
  xt::xtensor<double, 2> _foo;

  // Global indices as provided on Geometry creation
  std::vector<std::int64_t> _input_global_indices;
};

/// Build Geometry
/// @todo document
mesh::Geometry
create_geometry(MPI_Comm comm, const Topology& topology,
                const fem::CoordinateElement& coordinate_element,
                const graph::AdjacencyList<std::int64_t>& cells,
                const xt::xtensor<double, 2>& x,
                const std::function<std::vector<int>(
                    const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
                = nullptr);

} // namespace dolfinx::mesh
