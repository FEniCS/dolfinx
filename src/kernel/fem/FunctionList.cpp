#include <iostream>
#include <dolfin/FunctionList.h>

using namespace dolfin;

real zero (real x, real y, real z, real t) { return 0.0; }
real one  (real x, real y, real z, real t) { return 0.0; }

ShortList<FunctionList::FunctionData> FunctionList::list(DOLFIN_PARAMSIZE);
bool FunctionList::initialised = false;

//-----------------------------------------------------------------------------
int FunctionList::add(function f)
{
  // Make sure that we add the functions zero and one first
  if ( !initialised )
	 init();
  
  int id = list.add(FunctionData(f));
  
  if ( id == -1 ) {
	 cout << "ojoj" << endl;
	 list.resize(2*list.size());
	 id = list.add(FunctionData(f));
  }

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
real FunctionList::eval(int id, real x, real y, real z, real t)
{
  return list(id).f(x,y,z,t);
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
bool FunctionList::FunctionData::operator! ()
{
  return f == 0;
}
//-----------------------------------------------------------------------------
void FunctionList::FunctionData::operator= (int i)
{
  // FIXME: Use logging system
  if ( i != 0 ) {
	 cout << "Assignment to function pointer can only be 0." << endl;
	 exit(1);
  }
  
  f = 0;
}
//-----------------------------------------------------------------------------
