// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Map.h>
#include <dolfin/FunctionList.h>

namespace dolfin {
  real zero (real x, real y, real z, real t) { return 0.0; }
  real one  (real x, real y, real z, real t) { return 1.0; }
}

using namespace dolfin;

// Initialise static data
Array<FunctionList::FunctionData> FunctionList::list(DOLFIN_PARAMSIZE);
int FunctionList::_size = 0;
bool FunctionList::initialised = false;
FunctionList functionList;

//-----------------------------------------------------------------------------
FunctionList::FunctionList()
{
  if ( !initialised )
    init();
}
//-----------------------------------------------------------------------------
int FunctionList::add(function f)
{
  int id = list.add(FunctionData(f));
  
  if ( id == -1 ) {
    list.resize(2*list.size());
    id = list.add(FunctionData(f));
  }
  
  // Increase size of list. Note that _size <= list.size()
  _size += 1;
  
  return id;
}
//-----------------------------------------------------------------------------
void FunctionList::set(int id,
		       FunctionSpace::ElementFunction dX,
		       FunctionSpace::ElementFunction dY,
		       FunctionSpace::ElementFunction dZ,
		       FunctionSpace::ElementFunction dT)
{
  list(id).dX = dX;
  list(id).dY = dY;
  list(id).dZ = dZ;
  list(id).dT = dT;
}
//-----------------------------------------------------------------------------
void FunctionList::update(const FunctionSpace::ShapeFunction& v,
			  const Map& map)
{
  // FIXME: Possible optimisation is to use something like
  // map.dx(v, list(v.id()).dx) so that we avoid creating an
  // element function that needs to be copied

  list(v.id()).dx = map.dx(v);
  list(v.id()).dy = map.dy(v);
  list(v.id()).dz = map.dz(v);
  list(v.id()).dt = map.dt(v);
}
//-----------------------------------------------------------------------------
int FunctionList::size()
{
  return _size;
}
//-----------------------------------------------------------------------------
real FunctionList::eval(int id, real x, real y, real z, real t)
{
  return list(id).f(x, y, z, t);
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dx(int id)
{
  return list(id).dx;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dy(int id)
{
  return list(id).dy;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dz(int id)
{
  return list(id).dz;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dt(int id)
{
  return list(id).dt;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dX(int id)
{
  return list(id).dX;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dY(int id)
{
  return list(id).dY;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dZ(int id)
{
  return list(id).dZ;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::dT(int id)
{
  return list(id).dT;
}
//-----------------------------------------------------------------------------
void FunctionList::init()
{
  list.add(FunctionData(zero));
  list.add(FunctionData(one));

  _size = 2;
  
  initialised = true;
}
//-----------------------------------------------------------------------------
// FunctionList::FunctionData
//-----------------------------------------------------------------------------
FunctionList::FunctionData::FunctionData()
{
  f = 0;
}
//-----------------------------------------------------------------------------
FunctionList::FunctionData::FunctionData(function f)
{
  this->f = f;
}
//-----------------------------------------------------------------------------
void FunctionList::FunctionData::operator= (int zero)
{
  if ( zero != 0 )
    dolfin_error("Assignment to int must be zero.");
  
  f = 0;
}
//-----------------------------------------------------------------------------
bool FunctionList::FunctionData::operator! () const
{
  return f == 0;
}
//-----------------------------------------------------------------------------
