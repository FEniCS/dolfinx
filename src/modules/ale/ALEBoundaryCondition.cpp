#include <dolfin/ALEBoundaryCondition.h> 


using namespace dolfin;

//-----------------------------------------------------------------------------
ALEBoundaryCondition::ALEBoundaryCondition()
  : ref_coords_set(false), ref_ndx(0) { } 
//-----------------------------------------------------------------------------
ALEBoundaryCondition::~ALEBoundaryCondition()
{
  delete w;
}
//-----------------------------------------------------------------------------
void ALEBoundaryCondition::setBoundaryVelocity(ALEFunction& mesh_vel) { 
  w = &mesh_vel;
}
//-----------------------------------------------------------------------------
void ALEBoundaryCondition::endRecording() 
{
  ref_coords_set = true;
}
  
