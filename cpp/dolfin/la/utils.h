// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <petscis.h>
#include <petscvec.h>
#include <string>
#include <vector>

namespace dolfin
{
namespace common
{
class IndexMap;
}
namespace la
{

/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

/// Compute IndexSets (IS) for stacked index maps
std::vector<IS> compute_index_sets(std::vector<const common::IndexMap*> maps);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);

class VecWrapper
{
public:
  VecWrapper(Vec y, bool ghosted = true);
  ~VecWrapper();
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar* array = nullptr;
  Vec _y;
  Vec _y_local = nullptr;
};

class VecReadWrapper
{
public:
  VecReadWrapper(const Vec y, bool ghosted = true);
  ~VecReadWrapper();
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x;

private:
  PetscScalar const* array = nullptr;
  const Vec _y;
  Vec _y_local = nullptr;
};

} // namespace la
} // namespace dolfin
