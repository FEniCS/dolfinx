#include <iostream>
#include <dolfin/FunctionList.h>

using namespace dolfin;

real zero (real x, real y, real z, real t) { return 0.0; }
real one  (real x, real y, real z, real t) { return 1.0; }

// Initialise static data
ShortList<FunctionList::FunctionData> FunctionList::list(DOLFIN_PARAMSIZE);
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
							  FunctionSpace::ElementFunction dx,
							  FunctionSpace::ElementFunction dy,
							  FunctionSpace::ElementFunction dz,
							  FunctionSpace::ElementFunction dt)
{
  list(id).dx = dx;
  list(id).dy = dy;
  list(id).dz = dz;
  list(id).dt = dt;
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
FunctionSpace::ElementFunction FunctionList::dx(int id)
{
  return list(id).dx;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionList::dy(int id)
{
  return list(id).dy;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionList::dz(int id)
{
  return list(id).dz;
}
//-----------------------------------------------------------------------------
FunctionSpace::ElementFunction FunctionList::dt(int id)
{
  return list(id).dt;
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
  // FIXME: Use logging system
  if ( zero != 0 ) {
	 cout << "Assignment to int must be zero." << endl;
	 exit(1);
  }
  
  f = 0;
}
//-----------------------------------------------------------------------------
bool FunctionList::FunctionData::operator! () const
{
  return f == 0;
}
//-----------------------------------------------------------------------------
