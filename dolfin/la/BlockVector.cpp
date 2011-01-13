// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-08-25
// Last changed: 2011-01-12
//
// Modified by Anders Logg, 2008.

#include <algorithm>
#include <cmath>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/utils.h>
#include "GenericVector.h"
#include "Vector.h"
#include "BlockVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockVector::BlockVector(uint n) : vectors(n)
{
  for (uint i = 0; i < n; i++)
    vectors[i].reset(new Vector());
}
//-----------------------------------------------------------------------------
BlockVector::~BlockVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BlockVector* BlockVector::copy() const
{
  error("BlockVector needs a cleanup.");

  //BlockVector* x = new BlockVector(vectors.size());
  //for (uint i = 0; i < vectors.size(); i++)
  //  x->set(i, boost::shared_ptr<GenericVector>(vectors[i]->copy())));
  //return x;
  return 0;
}
//-----------------------------------------------------------------------------
GenericVector& BlockVector::operator()(uint i)
{
  assert(i < vectors.size());
  assert(vectors[i]);
  return *vectors[i];
}
//-----------------------------------------------------------------------------
dolfin::uint BlockVector::size() const
{
  return vectors.size();
}
//-----------------------------------------------------------------------------
void BlockVector::axpy(double a, const BlockVector& x)
{
  for (uint i = 0; i < vectors.size(); i++)
    vectors[i]->axpy(a, x.get(i));
}
//-----------------------------------------------------------------------------
double BlockVector::inner(const BlockVector& x) const
{
  double value = 0.0;
  for (uint i = 0; i < vectors.size(); i++)
    value += vectors[i]->inner(x.get(i));
  return value;
}
//-----------------------------------------------------------------------------
double BlockVector::norm(std::string norm_type) const
{
  double value = 0.0;
  if(norm_type == "l1")
  {
    for (uint i = 0; i < vectors.size(); i++)
      value += vectors[i]->norm(norm_type);
  }
  else if (norm_type == "l2")
  {
    for (uint i = 0; i < vectors.size(); i++)
      value += std::pow(vectors[i]->norm(norm_type), 2);
    value = sqrt(value);
  }
  else if (norm_type == "linf")
  {
    std::vector<double> linf(vectors.size());
    for (uint i = 0; i < vectors.size(); i++)
      linf[i] = vectors[i]->norm(norm_type);
     value = *(std::max_element(linf.begin(), linf.end()));
  }
  return value;
}
//-----------------------------------------------------------------------------
double BlockVector::min() const
{
  std::vector<double> _min(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    _min[i] = vectors[i]->min();

  return *(std::min_element(_min.begin(), _min.end()));
}
//-----------------------------------------------------------------------------
double BlockVector::max() const
{
  std::vector<double> _max(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    _max[i] = vectors[i]->min();

  return *(std::max_element(_max.begin(), _max.end()));
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator*= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    *vectors[i] *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator/= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    *vectors[i] /= a;
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator+= (const BlockVector& y)
{
  axpy(1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator-= (const BlockVector& y)
{
  axpy(-1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (const BlockVector& x)
{
  for(uint i = 0; i < vectors.size(); i++)
    *vectors[i] = x.get(i);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    *vectors[i] = a;
  return *this;
}
//-----------------------------------------------------------------------------
std::string BlockVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < vectors.size(); i++)
    {
      s << "  BlockVector " << i << std::endl << std::endl;
      s << indent(indent(vectors[i]->str(true))) << std::endl;
    }
  }
  else
  {
    s << "<BlockVector containing " << vectors.size() << " blocks>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void BlockVector::set(uint i, GenericVector& v)
{
  assert(i < vectors.size());

  // FIXME: Resolve copy/view approach
  vectors[i] = boost::shared_ptr<GenericVector>(reference_to_no_delete_pointer(v));
}
//-----------------------------------------------------------------------------
const GenericVector& BlockVector::get(uint i) const
{
  assert(i < vectors.size());
  return *(vectors[i]);
}
//-----------------------------------------------------------------------------
GenericVector& BlockVector::get(uint i)
{
  assert(i < vectors.size());
  return *(vectors[i]);
}
//-----------------------------------------------------------------------------
