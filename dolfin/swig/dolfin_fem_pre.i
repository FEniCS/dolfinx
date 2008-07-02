// Rename assemble --> cpp_assemble (gets mapped in assembly.py)
%rename(cpp_assemble) dolfin::assemble;
%rename(cpp_assemble_system) dolfin::assemble_system;
%rename(cpp_DirichletBC) dolfin::DirichletBC;
%rename(cpp_PeriodicBC) dolfin::PeriodicBC;
%rename(sub) dolfin::DofMapSet::operator[];

