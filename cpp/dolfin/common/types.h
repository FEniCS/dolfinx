// Copyright (C) 2008-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <petscsys.h>

// Typedefs for ufc_scalar
#ifdef PETSC_USE_COMPLEX
#include <complex>
using ufc_scalar_t = std::complex<double>;
#else
using ufc_scalar_t = double;
#endif
