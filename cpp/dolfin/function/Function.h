// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/la/PETScVector.h>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <vector>

namespace dolfin
{

namespace geometry
{
class BoundingBoxTree;
}
namespace mesh
{
class Mesh;
} // namespace mesh

namespace function
{
class FunctionSpace;

/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[     u_h = \sum_{i=1}^{n} U_i \phi_i \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.

class Function
{
public:
  /// Create function on given function space
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace> V);

  /// Create function on given function space with a given vector
  ///
  /// *Warning: This constructor is intended for internal library use only*
  ///
  /// @param[in] V The function space
  /// @param[in] x The vector
  Function(std::shared_ptr<const FunctionSpace> V, Vec x);

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  virtual ~Function() = default;

  /// Move assignment
  Function& operator=(Function&& v) = default;

  // Assignment
  Function& operator=(const Function& v) = delete;

  /// Extract subfunction (view into the Function)
  /// @param[i] i Index of subfunction
  /// @return The subfunction
  Function sub(int i) const;

  /// Collapse a subfunction (view into the Function) to a stand-alone
  /// Function
  Function collapse() const;

  /// Return shared pointer to function space
  /// @return The function space
  std::shared_ptr<const FunctionSpace> function_space() const;

  /// Return vector of expansion coefficients (non-const version)
  /// @return The vector of expansion coefficients
  la::PETScVector& vector();

  /// Return vector of expansion coefficients (const version)
  /// @return The vector of expansion coefficients
  const la::PETScVector& vector() const;

  /// Interpolate a Function (on possibly non-matching meshes)
  /// @param[in] v The function to be interpolated.
  void interpolate(const Function& v);

  /// Interpolate expression
  /// @param[in] f The expression to be interpolated.
  void interpolate(
      const std::function<void(
          Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>,
          const Eigen::Ref<const Eigen::Array<
              double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>)>& f);

  /// Return value rank
  int value_rank() const;

  /// Return value size
  int value_size() const;

  /// Return value dimension for given axis
  /// @param[in] i The index of the axis
  /// @returns The value dimension.
  int value_dimension(int i) const;

  /// Return value shape
  std::vector<int> value_shape() const;

  /// Evaluate at given point in given cell
  /// @param[in] x The coordinates of the points
  /// @param[in] cell The cell which contains the given point
  /// @param[in,out] u The values at the points
  void
  eval(const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
           x,
       const mesh::MeshEntity& cell,
       Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           u) const;

  /// Evaluate function at given coordinates
  /// @param[in] x The coordinates of the points
  /// @param[in] bb_tree Bounding box tree for the mesh
  /// @param[in,out] u The values at the points
  void
  eval(const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::RowMajor>>
           x,
       const geometry::BoundingBoxTree& bb_tree,
       Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
           u) const;

  /// Restrict function to local cell (compute expansion coefficients w)
  /// @param[in] cell The cell
  /// @param[in] coordinate_dofs The coordinate dofs
  /// @param[in,out] w Expansion coefficients.
  void restrict(const mesh::MeshEntity& cell,
                const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs,
                PetscScalar* w) const;

  /// Compute values at all mesh points
  /// @return The values at all geometric points
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  compute_point_values() const;

  /// Name
  std::string name = "u";

  /// ID
  const std::size_t id;

private:
  // The function space
  std::shared_ptr<const FunctionSpace> _function_space;

  // The vector of expansion coefficients (local)
  la::PETScVector _vector;
};
} // namespace function
} // namespace dolfin
