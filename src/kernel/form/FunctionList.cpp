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
		       FunctionSpace::ElementFunction ddX,
		       FunctionSpace::ElementFunction ddY,
		       FunctionSpace::ElementFunction ddZ,
		       FunctionSpace::ElementFunction ddT)
{
  list(id).ddX = ddX;
  list(id).ddY = ddY;
  list(id).ddZ = ddZ;
  list(id).ddT = ddT;
}
//-----------------------------------------------------------------------------
void FunctionList::update(const FunctionSpace::ShapeFunction& v,
			  const Map& map)
{
  // FIXME: Possible optimisation is to use something like
  // map.ddx(v, list(v.id()).ddx) so that we avoid creating an
  // element function that needs to be copied

  list(v.id()).ddx = map.ddx(v);
  list(v.id()).ddy = map.ddy(v);
  list(v.id()).ddz = map.ddz(v);
  list(v.id()).ddt = map.ddt(v);
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
const FunctionSpace::ElementFunction& FunctionList::ddx(int id)
{
  return list(id).ddx;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddy(int id)
{
  return list(id).ddy;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddz(int id)
{
  return list(id).ddz;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddt(int id)
{
  return list(id).ddt;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddX(int id)
{
  return list(id).ddX;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddY(int id)
{
  return list(id).ddY;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddZ(int id)
{
  return list(id).ddZ;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction& FunctionList::ddT(int id)
{
  return list(id).ddT;
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
