function xmlmesh(filename,p,t)

% XMLMESH - SAVE MATLAB 2D (AND FEMLAB 3D) MESH TO DOLFIN XML FORMAT
%
% Usage: xmlmesh(filename,p,t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)
%
% Copyright (C) 2004 Erik Svensson.
% Licensed under the GNU GPL Version 2.
%
% Modified by Anders Logg 2004-2005.
%
% First added:  2004-02-10
% Last changed: 2005

% Open file
fp = fopen(filename,'w');

np = size(p,2);
nt = size(t,2);

% Write header
fprintf(fp,'<?xml version="1.0" encoding="UTF-8"?>\n\n');
fprintf(fp,'<dolfin xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">\n');

% 2D mesh
if (size(p,1) == 2)

  % Write nodes
  disp('Writing nodes...')
  fprintf(fp,'  <mesh>\n');
  fprintf(fp,'    <nodes size="%d">\n',np);  
  for n=1:np
    fprintf(fp,'      <node name="%d" x="%f" y="%f" z="0.0"/>\n', ...
	    n-1, p(1,n), p(2,n));
  end
  fprintf(fp,'    </nodes>\n');
  
  % Write cells
  disp('Writing cells...')
  fprintf(fp,'    <cells size="%d">\n',nt);
  for n=1:nt
    fprintf(fp,'      <triangle name="%d" n0="%d" n1="%d" n2="%d"/>\n', ...
	    n-1,t(1,n)-1,t(2,n)-1,t(3,n)-1);
  end
  fprintf(fp,'    </cells>\n');
  fprintf(fp,'  </mesh>\n');
  fprintf(fp,'</dolfin>\n');  
  
% 3D mesh
elseif (size(p,1) == 3)

  % Write nodes
  disp('Writing nodes...')
  fprintf(fp,'  <mesh>\n');
  fprintf(fp,'    <nodes size="%d">\n',np);  
  for n=1:np
    fprintf(fp,'      <node name="%d" x="%f" y="%f" z="%f"/>\n', ...
            n-1,p(1,n),p(2,n),p(3,n));
  end
  fprintf(fp,'    </nodes>\n');
  
  % Write cells
  disp('Writing cells...')
  fprintf(fp,'    <cells size="%d">\n',nt);
  for n=1:nt
    fprintf(fp,'      <tetrahedron name="%d" n0="%d" n1="%d" n2="%d" n3="%d"/>\n', ...
            n-1,t(1,n)-1,t(2,n)-1,t(3,n)-1,t(4,n)-1);
  end
  fprintf(fp,'    </cells>\n');
  fprintf(fp,'  </mesh>\n');
  fprintf(fp,'</dolfin>\n');
  
end

% Close file
fclose(fp);
disp('Done')
