// Copyright (C) 2008 Kent-Andre Mardal
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
// Modified by Garth N. Wells, 2011.
//
// First added:  2008-08-25
// Last changed: 2011-01-22
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
  BlockVector* x = new BlockVector(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    x->set_block(i, boost::shared_ptr<GenericVector>(vectors[i]->copy()));
  return x;
}
//-----------------------------------------------------------------------------
void BlockVector::set_block(uint i, boost::shared_ptr<GenericVector> v)
{
  dolfin_assert(i < vectors.size());
  vectors[i] = v;
}
//-----------------------------------------------------------------------------
const boost::shared_ptr<GenericVector> BlockVector::get_block(uint i) const
{
  dolfin_assert(i < vectors.size());
  return vectors[i];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> BlockVector::get_block(uint i)
{
  dolfin_assert(i < vectors.size());
  return vectors[i];
}
//-----------------------------------------------------------------------------
bool BlockVector::empty() const
{
  return vectors.empty();
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
    vectors[i]->axpy(a, *x.get_block(i));
}
//-----------------------------------------------------------------------------
double BlockVector::inner(const BlockVector& x) const
{
  double value = 0.0;
  for (uint i = 0; i < vectors.size(); i++)
    value += vectors[i]->inner(*x.get_block(i));
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
    *vectors[i] = *x.get_block(i);
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
