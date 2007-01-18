#ifndef __ALE_BOUNDARYCONDITION_H
#define __ALE_BOUNDARYCONDITION_H

#include <vector>

#include <dolfin/BoundaryCondition.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/Function.h>
#include <dolfin/Point.h>

#include <dolfin/ALEFunction.h>


namespace dolfin
{
  
  /// ALEBoundaryCondition allows the user to set boundary conditions
  /// using the reference coordinates as well as the spacial coordinates.
  
  class ALEBoundaryCondition : public BoundaryCondition
  {
    
  public :
    
    /// Constructor
    ALEBoundaryCondition();
    
    /// Destructor
    virtual ~ALEBoundaryCondition();
    
    /// BoundaryCondition eval function
    virtual void eval(BoundaryValue& value, const Point& p, unsigned int i) {
      if (!ref_coords_set)
	ref.push_back(p);
      else 
	ref_ndx = ref_ndx % ref.size();
      
      eval(value, p, ref[ref_ndx++], i); 
    }
    
    /// BoundaryCondition eval function using also reference coordinates
    virtual void eval(BoundaryValue& value, 
		      const Point& p, 
		      const Point& r, 
		      unsigned int i) = 0;
    
    /// Specify the mesh velocity function
    void setBoundaryVelocity(ALEFunction& mesh_vel);
    
    /// End recording reference coords
    void endRecording();
    
    
  protected :
    
    /// Mesh boundary velocity function
    ALEFunction* w;
    
    
  private :
    
    /// Vector list of material coordinate points on the boundary
    std::vector<Point> ref;
    
    /// true if this has already recorded all reference frame points
    bool ref_coords_set;
    
    /// reference coordinate index
    unsigned int ref_ndx;
    
  };
  
}


    
#endif
