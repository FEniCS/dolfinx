// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <petscvec.h>

namespace dolfinx::function::detail
{
Vec create_ghosted_vector(
    const common::IndexMap& map,
    const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& x)
{
  const int bs = map.block_size();
  std::int32_t size_local = bs * map.size_local();
  std::int32_t num_ghosts = bs * map.num_ghosts();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map.ghosts();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _ghosts(bs * ghosts.rows());
  for (int i = 0; i < ghosts.rows(); ++i)
  {
    for (int j = 0; j < bs; ++j)
      _ghosts[i * bs + j] = bs * ghosts[i] + j;
  }

  Vec vec;
  VecCreateGhostWithArray(map.comm(), size_local, PETSC_DECIDE, num_ghosts,
                          _ghosts.data(), x.array().data(), &vec);
  return vec;
}
} // namespace dolfinx::function::detail
