// Rename assemble --> cpp_assemble (gets mapped in assembly.py)
%rename(cpp_assemble) dolfin::assemble;
%rename(cpp_DirichletBC) dolfin::DirichletBC;
%rename(sub) dolfin::DofMapSet::operator[];

