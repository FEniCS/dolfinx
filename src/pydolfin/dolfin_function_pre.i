// Rename Function --> cpp_Function (gets mapped in assembly.py)
%rename(cpp_Function) dolfin::Function;
%rename(sub) dolfin::Function::operator[];
%rename(cpp_FacetNormal) dolfin::FacetNormal;
%rename(cpp_MeshSize) dolfin::MeshSize;
%rename(cpp_AvgMeshSize) dolfin::AvgMeshSize;
