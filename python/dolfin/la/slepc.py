# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


class SLEPcEigenSolver:
    def __init__(self, mpi_comm):
        self._cpp_object = cpp.la.SLEPcEigenSolver(mpi_comm)

    def set_options_prefix(self, opt_prefix: str):
        self._cpp_object.set_options_prefix(opt_prefix)

    def set_from_options(self):
        self._cpp_object.set_from_options()

    def set_operators(self, A, B):
        self._cpp_object.set_operators(A, B)

    def get_options_prefix(self):
        return self._cpp_object.get_options_prefix()

    def get_number_converged(self):
        return self._cpp_object.get_number_converged()

    def set_deflation_space(self, space):
        self._cpp_object.set_deflation_space(space)

    def set_initial_space(self, space):
        self._cpp_object.set_initial_space(space)

    def solve(self):
        """Compute all eigenpairs"""
        self._cpp_object.solve()

    def solve_part(self, n: int):
        """Compute first n eigenpairs"""
        self._cpp_object.solve(n)

    def get_eigenvalue(self, i):
        """Get i-th eigenvalues"""
        return self._cpp_object.get_eigenvalue(i)

    def get_eigenpair(self, i):
        """Get i-th eigenpair

        Returns
        -------
        Tuple of (eigenvalue real part, eigenvalue complex part,
        eigenvector real part, eigenvector complex part)
        """
        return self._cpp_object.get_eigenpair(i)
