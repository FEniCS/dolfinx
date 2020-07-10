// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Function.h"
#include "FunctionSpace.h"
#include <Eigen/Dense>
#include <functional>

namespace dolfinx::function
{

/// Interpolate a Function (on possibly non-matching meshes)
/// @param[in,out] u The function to interpolate into
/// @param[in] v The function to be interpolated
void interpolate(Function& u, const Function& v)
{
  u.function_space()->interpolate(u.x()->array(), v);
}

/// Interpolate an expression
/// @param[in,out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
void interpolate(
    Function& u,
    const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& f)
{
  u.function_space()->interpolate(u.x()->array(), f);
}

/// Interpolate an expression. This interface uses an expression
/// function f that has an in/out argument for the expression values.
/// It is primarily to support C code implementations of the
/// expression, e.g. using Numba. Generally the interface where the
/// expression function is a pure function, i.e. the expression values
/// are the return argument, should be preferred.
/// @param[in,out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
void interpolate_c(
    Function& u,
    const std::function<
        void(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>>,
             const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                                 Eigen::RowMajor>>&)>& f)
{
  u.function_space()->interpolate_c(u.x()->array(), f);
}

} // namespace dolfinx::function
