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
  BlockVector* x= new BlockVector(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    x->set(i, *(this->get(i).copy()));
  return x;
}
//-----------------------------------------------------------------------------
SubVector BlockVector::operator()(uint i)
{
  SubVector sv(i, *this);
  return sv;
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
    this->get(i).axpy(a, x.get(i));
}
//-----------------------------------------------------------------------------
double BlockVector::inner(const BlockVector& x) const
{
  double value = 0.0;
  for (uint i = 0; i < vectors.size(); i++)
    value += this->get(i).inner(x.get(i));
  return value;
}
//-----------------------------------------------------------------------------
double BlockVector::norm(std::string norm_type) const
{
  double value = 0.0;
  if(norm_type == "l1")
  {
    for (uint i = 0; i < vectors.size(); i++)
      value += this->get(i).norm(norm_type);
  }
  else if (norm_type == "l2")
  {
    for (uint i = 0; i < vectors.size(); i++)
      value += std::pow(this->get(i).norm(norm_type), 2);
    value = sqrt(value);
  }
  else if (norm_type == "linf")
  {
    std::vector<double> linf(vectors.size());
    for (uint i = 0; i < vectors.size(); i++)
      linf[i] = this->get(i).norm(norm_type);
     value = *(std::max_element(linf.begin(), linf.end()));
  }
  return value;
}
//-----------------------------------------------------------------------------
double BlockVector::min() const
{
  std::vector<double> _min(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    _min[i] = this->get(i).min();

  return *(std::min_element(_min.begin(), _min.end()));
}
//-----------------------------------------------------------------------------
double BlockVector::max() const
{
  std::vector<double> _max(vectors.size());
  for (uint i = 0; i < vectors.size(); i++)
    _max[i] = this->get(i).min();

  return *(std::max_element(_max.begin(), _max.end()));
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator*= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    this->get(i) *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator/= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    this->get(i) /= a;
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
    this->get(i) = x.get(i);
  return *this;
}
//-----------------------------------------------------------------------------
const BlockVector& BlockVector::operator= (double a)
{
  for(uint i = 0; i < vectors.size(); i++)
    this->get(i) = a;
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
      s << indent(indent(get(i).str(true))) << std::endl;
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
  error("BlockVector::set needs to be updated");
  //matrices[i*n+j] = m.copy(); //FIXME. not obvious that copy is the right thing
  //vectors[i] = &v; //FIXME. not obvious that copy is the right thing
}
//-----------------------------------------------------------------------------
const GenericVector& BlockVector::get(uint i) const
{
  return *(vectors[i]);
}
//-----------------------------------------------------------------------------
GenericVector& BlockVector::get(uint i)
{
  return *(vectors[i]);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
SubVector::SubVector(uint n, BlockVector& bv) : n(n), bv(bv)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubVector::~SubVector()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const SubVector& SubVector::operator=(GenericVector& v)
{
  bv.set(n, v);
  return *this;
}
//-----------------------------------------------------------------------------
