#ifndef __MTL4_H
#define __MTL4_H

#ifdef HAS_MTL4
#include <boost/numeric/mtl/mtl.hpp>
namespace dolfin
{
  typedef mtl::compressed2D<double> MTL4_sparse_matrix;
  typedef mtl::dense_vector<double> MTL4_vector;
}
#endif
#endif


