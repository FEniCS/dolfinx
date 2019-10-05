// Copyright (C) 2008-2014 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <petscsys.h>

// Typedefs for ufc_scalar
#ifdef PETSC_USE_COMPLEX
#include <complex>
using ufc_scalar_t = std::complex<double>;
#else
using ufc_scalar_t = double;
#endif

namespace dolfin
{

// Typedefs for Eigen

// double Arrays
using EigenArrayXd = Eigen::Array<double, Eigen::Dynamic, 1>;
using EigenRowArrayXd = Eigen::Array<double, 1, Eigen::Dynamic>;
using EigenRowArrayXXd
    = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// int64 Arrays
using EigenRowArrayXXi64 = Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>;

} // namespace dolfin
