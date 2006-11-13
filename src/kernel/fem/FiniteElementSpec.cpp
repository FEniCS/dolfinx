// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-19
// Last changed: 2006-02-19

#include <sstream>

#include <dolfin/constants.h>
#include <dolfin/FiniteElementSpec.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FiniteElementSpec::FiniteElementSpec()
{
  init("unspecified", "unspecified", 0, 0);
}
//-----------------------------------------------------------------------------
FiniteElementSpec::FiniteElementSpec(std::string type)
{
  init(type, "unspecified", 0, 0);
}
//-----------------------------------------------------------------------------
FiniteElementSpec::FiniteElementSpec(std::string type, std::string shape,
				     uint degree, uint vectordim)
{
  init(type, shape, degree, vectordim);
}
//-----------------------------------------------------------------------------
void FiniteElementSpec::init(std::string type, std::string shape,
			     uint degree, uint vectordim)
{
  _type = type;
  _shape = shape;
  _degree = degree;
  _vectordim = vectordim;
}
//-----------------------------------------------------------------------------
std::string FiniteElementSpec::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
std::string FiniteElementSpec::shape() const
{
  return _shape;
}
//-----------------------------------------------------------------------------
dolfin::uint FiniteElementSpec::degree() const
{
  return _degree;
}
//-----------------------------------------------------------------------------
dolfin::uint FiniteElementSpec::vectordim() const
{
  return _vectordim;
}
//-----------------------------------------------------------------------------
std::string FiniteElementSpec::repr() const
{
  std::ostringstream stream;
  
  if ( _vectordim > 0 )
  {
    stream << "[ " << _type << " finite element of degree " << _degree
	   << " on a " << _shape << " with " << _vectordim << " ]";
  }
  else
  {
    stream << "[ " << _type << " finite element of degree " << _degree
	   << " on a " << _shape << " ]";
  }

  return stream.str();
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const FiniteElementSpec& spec)
{
  stream << spec.repr();

  return stream;
}
//-----------------------------------------------------------------------------
