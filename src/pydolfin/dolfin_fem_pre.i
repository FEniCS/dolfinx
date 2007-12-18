// Rename assemble --> cpp_assemble (gets mapped in assembly.py)
%rename(cpp_assemble) dolfin::assemble;
%rename(sub) dolfin::DofMapSet::operator[];

