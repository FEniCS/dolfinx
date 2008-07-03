// Rename Function --> cpp_Function (gets mapped in assembly.py)
%rename(cpp_Function) dolfin::Function;
%rename(sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(cpp_FacetNormal) dolfin::FacetNormal;
%rename(cpp_MeshSize) dolfin::MeshSize;
%rename(cpp_AvgMeshSize) dolfin::AvgMeshSize;
%rename(cpp_FacetArea) dolfin::FacetArea;
%rename(cpp_InvFacetArea) dolfin::InvFacetArea;

// Trick to expose protected member cell() in Python
%rename(old_cell) dolfin::Function::cell;
%rename(cell) dolfin::Function::new_cell;

// Trick to expose protected member normal() in Python
%rename(old_normal) dolfin::Function::normal;
%rename(normal) dolfin::Function::new_normal;

// Assign typemap for constant vector function constructor
// TODO: Make (size,values) typemap work, preferably supporting both arrays and sequences (tuples):
//%typemap(in) (dolfin::uint size, const dolfin::real* values) = (int _array_dim, double* _array);
%typemap(in) (const dolfin::real* values) = (double* _array);

