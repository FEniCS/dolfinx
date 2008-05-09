// Rename Function --> cpp_Function (gets mapped in assembly.py)
%rename(cpp_Function) dolfin::Function;
%rename(sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(cpp_FacetNormal) dolfin::FacetNormal;
%rename(cpp_MeshSize) dolfin::MeshSize;
%rename(cpp_AvgMeshSize) dolfin::AvgMeshSize;

// Trick to expose protected member cell() in Python
%rename(old_cell) dolfin::Function::cell;
%rename(cell) dolfin::Function::new_cell;

// Trick to expose protected member normal() in Python
%rename(old_normal) dolfin::Function::normal;
%rename(normal) dolfin::Function::new_normal;
