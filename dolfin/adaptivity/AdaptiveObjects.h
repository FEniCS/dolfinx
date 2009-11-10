// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-09
// Last changed: 2009-11-10

#ifndef __ADAPTIVE_OBJECTS
#define __ADAPTIVE_OBJECTS

#include <set>
#include <map>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class FunctionSpace;
  class Function;
  class BoundaryCondition;
  template <class T> class MeshFunction;

  /// This class handles the automatic update/refinement of adaptive
  /// objects when meshes are refined. It is a singleton object that
  /// stores a forest (set of trees) where the root node of each tree
  /// is a mesh, and the leaves are function spaces, functions and
  /// boundary conditions.

  class AdaptiveObjects
  {
  public:

    //--- Registration of objects ---

    /// Register function space
    static void register_object(FunctionSpace* function_space);

    /// Register function
    static void register_object(Function* function);

    /// Register boundary condition
    static void register_object(BoundaryCondition* boundary_condition);

    //--- Deregistration of objects ---

    /// Deregister function space
    static void deregister_object(FunctionSpace* function_space);

    /// Deregister function
    static void deregister_object(Function* function);

    /// Deregister boundary condition
    static void deregister_object(BoundaryCondition* boundary_condition);

    //--- Refinement ---

    /// Refine mesh
    static void refine(Mesh* mesh,
                       MeshFunction<bool>* cell_markers);

    /// Refine function space to new mesh
    static void refine(FunctionSpace* function_space,
                       Mesh& new_mesh);

    /// Refine function to new function space
    static void refine(Function* function,
                       FunctionSpace& new_function_space);

    /// Refine boundary condition to new function space
    static void refine(BoundaryCondition* boundary_condition,
                       FunctionSpace& new_function_space);

  private:

    // Constructor is private
    AdaptiveObjects() {}

    // Destructor
    ~AdaptiveObjects() {}

    // Singleton object
    static AdaptiveObjects objects;

    // Mapping from meshes to function spaces
    std::map<const Mesh*, std::set<FunctionSpace*> > _function_spaces;

    // Mapping from function spaces to functions
    std::map<const FunctionSpace*, std::set<Function*> > _functions;

    // Mapping from function spaces to boundary conditions
    std::map<const FunctionSpace*, std::set<BoundaryCondition*> > _boundary_conditions;

  };

}

#endif
