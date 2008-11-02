// Import ufc.h so ufc::function is known
%import ufc.h

// Rename misc function classes from Foo --> cpp_Foo (gets mapped in function.py)
%rename(cpp_Function) dolfin::Function;
%rename(cpp_FacetNormal) dolfin::FacetNormal;
%rename(cpp_MeshSize) dolfin::MeshSize;
%rename(cpp_AvgMeshSize) dolfin::AvgMeshSize;
%rename(cpp_FacetArea) dolfin::FacetArea;
%rename(cpp_InvFacetArea) dolfin::InvFacetArea;

// Modifying the interface of Function
%rename(sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

// Trick to expose protected member cell() in Python
%rename(old_cell) dolfin::Function::cell;
%rename(cell) dolfin::Function::new_cell;

// Trick to expose protected member normal() in Python
%rename(old_normal) dolfin::Function::normal;
%rename(normal) dolfin::Function::new_normal;



