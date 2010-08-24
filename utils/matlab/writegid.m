function writegid(p,t,filename)

% WRITEGID - SAVE GRID TO INP FORMAT
%
% Usage: writegid(p,t,filename)
%
% Saves matlab grid to GiD mesh file.
% Output file name is filename.flavia.msh.
% Only for two-dimensional triangular meshes.
% See 'help initmesh' for p and t.
%
% (c) 2002 by
%
%   Rasmus Hemph       (PDE project course 2001/2002)
%   Alexandra Krusper  (PDE project course 2001/2002)
%   Walter Villanueva  (PDE project course 2001/2002)
%
% Distributed under the GNU LGPL Version 2.1.
%
% Slightly modified by Hoffman/Logg.
 
filename=strcat(filename,'.flavia.msh');
fid = fopen(filename,'w');
  
fprintf(fid,'mesh dimension = 2 elemtype triangle nnode = 3\n');
fprintf(fid,'coordinates\n');
for(i=1:length(P))
  fprintf(fid, '%i %f %f 0.0\n', i, P(1,i), P(2,i));
end
fprintf(fid,'end coordinates\n');
  
fprintf(fid,'elements\n');
  
for(i=1:length(T))
  fprintf(fid, '%i %d %d %d 1\n', i,T(1,i), T(2,i), T(3,i));
end
fprintf(fid,'end elements\n');
fclose(fid);
