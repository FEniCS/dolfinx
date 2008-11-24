// Define classes in dolfin that are stored using shared_ptr
// Objects of these classes can then be passed to c++ functions
// demanding a shared_ptr<type>

# if defined(SWIG_SHARED_PTR_QNAMESPACE)
SWIG_SHARED_PTR(DofMap,dolfin::DofMap)
SWIG_SHARED_PTR(FiniteElement,dolfin::FiniteElement)

SWIG_SHARED_PTR(cpp_FunctionSpace,dolfin::FunctionSpace)
SWIG_SHARED_PTR_DERIVED(SubSpace,dolfin::FunctionSpace,dolfin::SubSpace)

SWIG_SHARED_PTR(cpp_Function,dolfin::Function)
SWIG_SHARED_PTR_DERIVED(Constant,dolfin::Function,dolfin::Constant)
SWIG_SHARED_PTR_DERIVED(cpp_MeshSize,dolfin::Function,dolfin::MeshSize)
SWIG_SHARED_PTR_DERIVED(cpp_AvgMeshSize,dolfin::Function,dolfin::AvgMeshSize)
SWIG_SHARED_PTR_DERIVED(cpp_InvMeshSize,dolfin::Function,dolfin::InvMeshSize)
SWIG_SHARED_PTR_DERIVED(cpp_FacetNormal,dolfin::Function,dolfin::FacetNormal)
SWIG_SHARED_PTR_DERIVED(cpp_FacetArea,dolfin::Function,dolfin::FacetArea)
SWIG_SHARED_PTR_DERIVED(cpp_InvFacetArea,dolfin::Function,dolfin::InvFacetArea)
SWIG_SHARED_PTR_DERIVED(OutflowFacet,dolfin::Function,dolfin::OutflowFacet)
SWIG_SHARED_PTR_DERIVED(cpp_DiscreteFunction,dolfin::Function,dolfin::cpp_DiscreteFunction)

SWIG_SHARED_PTR(Mesh,dolfin::Mesh)
SWIG_SHARED_PTR_DERIVED(BoundaryMesh,dolfin::Mesh,dolfin::BoundaryMesh)
SWIG_SHARED_PTR_DERIVED(UnitCube,dolfin::Mesh,dolfin::UnitCube)
SWIG_SHARED_PTR_DERIVED(UnitInterval,dolfin::Mesh,dolfin::UnitInterval)
SWIG_SHARED_PTR_DERIVED(Interval,dolfin::Mesh,dolfin::Interval)
SWIG_SHARED_PTR_DERIVED(UnitSquare,dolfin::Mesh,dolfin::UnitSquare)
SWIG_SHARED_PTR_DERIVED(UnitCircle,dolfin::Mesh,dolfin::UnitCircle)
SWIG_SHARED_PTR_DERIVED(Box,dolfin::Mesh,dolfin::Box)
SWIG_SHARED_PTR_DERIVED(Rectangle,dolfin::Mesh,dolfin::Rectangle)
SWIG_SHARED_PTR_DERIVED(UnitSphere,dolfin::Mesh,dolfin::UnitSphere)

// To be able to pass shared ufc objects we need to define these here and
// %include ufc.h
SWIG_SHARED_PTR(finite_element,ufc::finite_element)
SWIG_SHARED_PTR(dof_map,ufc::dof_map)
SWIG_SHARED_PTR(form,ufc::form)

%include ufc.h

#endif
