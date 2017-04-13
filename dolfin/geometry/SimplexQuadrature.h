// Copyright (C) 2014 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-02-24
// Last changed: 2017-04-07

#ifndef __SIMPLEX_QUADRATURE_H
#define __SIMPLEX_QUADRATURE_H

#include <vector>
#include <Eigen/Dense>
#include "Point.h"

namespace dolfin
{

  // Forward declarations
  class Cell;

  class SimplexQuadrature
  {
  public:

    /// Create SimplexQuadrature rules for reference simplex
    ///
    /// *Arguments*
    ///     tdim (std::size_t)
    ///         The topological dimension of the simplex.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    SimplexQuadrature(std::size_t tdim,
		      std::size_t order);

    /// Compute quadrature rule for cell.
    ///
    /// *Arguments*
    ///     cell (Cell)
    ///         The cell.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule(const Cell& cell,
			    std::size_t order) const;

    /// Compute quadrature rule for simplex.
    ///
    /// *Arguments*
    ///     coordinates (std::vector<Point>)
    ///         Vertex coordinates for the simplex
    ///     tdim (std::size_t)
    ///         The topological dimension of the simplex.
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule(const std::vector<Point>& coordinates,
			    std::size_t gdim,
			    std::size_t order) const;

    /// Compute quadrature rule for interval.
    ///
    /// *Arguments*
    ///     coordinates (std::vector<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_interval(const std::vector<Point>& coordinates,
				     std::size_t gdim,
				     std::size_t order) const;

    /// Compute quadrature rule for triangle.
    ///
    /// *Arguments*
    ///     coordinates (std::vector<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_triangle(const std::vector<Point>& coordinates,
				     std::size_t gdim,
				     std::size_t order) const;

    /// Compute quadrature rule for tetrahedron.
    ///
    /// *Arguments*
    ///     coordinates (std::vector<Point>)
    ///         Vertex coordinates for the simplex
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///     order (std::size_t)
    ///         The order of convergence of the quadrature rule.
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double>>
    ///         A flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    std::pair<std::vector<double>, std::vector<double>>
    compute_quadrature_rule_tetrahedron(const std::vector<Point>& coordinates,
					std::size_t gdim,
					std::size_t order) const;

    /// Compress a quadrature rule using algorithms from
    ///     Compression of multivariate discrete measures and applications
    ///     A. Sommariva, M. Vianello
    ///     Numerical Functional Analysis and Optimization
    ///     Volume 36, 2015 - Issue 9
    ///
    /// *Arguments*
    ///     qr (std::pair<std::vector<double>, std::vector<double>>)
    ///         The quadrature rule to be compressed
    ///     gdim (std::size_t)
    ///         The geometric dimension
    ///     quadrature_order (std::size_t)
    ///         The order of the quadrature rule
    ///
    /// *Returns*
    ///     std::vector<std::size_t>
    ///         The indices of the points that were kept (empty
    ///         if no compression was made)
    static std::vector<std::size_t>
    compress(std::pair<std::vector<double>, std::vector<double>>& qr,
	     std::size_t gdim,
	     std::size_t quadrature_order);

  private:

    // Setup quadrature rule on a reference simplex
    void setup_qr_reference_interval(std::size_t order);
    void setup_qr_reference_triangle(std::size_t order);
    void setup_qr_reference_tetrahedron(std::size_t order);


    // Utility function for computing a Vandermonde type matrix in a
    // Chebyshev basis
    static Eigen::MatrixXd
    Chebyshev_Vandermonde_matrix
    (const std::pair<std::vector<double>, std::vector<double>>& qr,
     std::size_t gdim,
     std::size_t N);

    // Utility function for computing a Chebyshev basis
    static std::vector<Eigen::VectorXd>
    Chebyshev_polynomial(const Eigen::VectorXd& x,
			 std::size_t N);

    // Utility function for creating a matrix with coefficients in
    // graded lexicographic order
    static std::vector<std::vector<std::size_t>>
    grlex(std::size_t gdim,
	  std::size_t N);

    // Utility function for calculating all combinations (n over k)
    static std::size_t
    choose(std::size_t n,
	   std::size_t k);

    // Compute Duanvant quadrature rules for triangle
    // Taken from
    //  https://people.sc.fsu.edu/~jburkardt/cpp_src/triangle_dunavant_rule/triangle_dunavant_rule.cpp
    static void dunavant_rule(std::size_t order,
			      std::vector<std::vector<double> >& p,
			      std::vector<double>& w);
    static std::size_t dunavant_order_num(std::size_t rule);
    static std::vector<std::size_t> dunavant_suborder(int rule, int suborder_num);
    static std::size_t dunavant_suborder_num(int rule);
    static void dunavant_subrule(std::size_t rule,
				 std::size_t suborder_num,
				 std::vector<double>& suborder_xyz,
				 std::vector<double>& w);
    static void dunavant_subrule_01(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_02(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_03(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_04(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_05(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_06(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_07(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_08(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_09(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_10(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_11(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_12(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_13(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_14(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_15(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_16(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_17(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_18(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_19(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static void dunavant_subrule_20(int suborder_num,
				    std::vector<double>& suborder_xyz,
				    std::vector<double>& suborder_w);
    static int i4_modp(int i, int j);
    static int i4_wrap(int ival, int ilo, int ihi);

    // Compute Gauss-Legendre quadrature rules for line
    // Taken from
    // https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule_fast/legendre_rule_fast.cpp
    static void legendre_compute_glr(std::size_t n,
				     std::vector<double>& x,
				     std::vector<double>& w);
    static void legendre_compute_glr0(std::size_t n,
				      double& p,
				      double& pp);
    static void legendre_compute_glr1(std::size_t n,
				      std::vector<double>& x,
				      std::vector<double>& w);
    static void legendre_compute_glr2(double pn0, int n, double& x1, double& d1);
    static double ts_mult(std::vector<double>& u, double h, int n);
    static double rk2_leg(double t1, double t2, double x, int n);

    // Quadrature rule on reference simplex (points and weights)
    std::vector<std::vector<double> > _p;
    std::vector<double> _w;

  };

}

#endif
