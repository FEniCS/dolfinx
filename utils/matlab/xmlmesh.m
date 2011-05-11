function xmlmesh(filename, p, t)

% XMLMESH - SAVE MATLAB 2D (AND FEMLAB 3D) MESH TO DOLFIN XML FORMAT
%
% Usage: xmlmesh(filename,p,t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)
%
% Copyright (C) 2004 Erik Svensson
%
% This file is part of DOLFIN.
%
% DOLFIN is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% DOLFIN is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
%
% Modified by Anders Logg 2004-2005.
% Modified by Marie Rognes 2009.
%
% First added:  2004-02-10
% Last changed: 2009-11-18

% Open file
fp = fopen(filename, 'w');
disp('Exporting mesh to DOLFIN XML format')

% Get number of points and triangles
np = size(p, 2);
nt = size(t, 2);

% Write header
fprintf(fp,'<?xml version="1.0" encoding="UTF-8"?>\n\n');
fprintf(fp,'<dolfin xmlns:dolfin="http://www.fenics.org/dolfin">\n');

% 2D mesh
if (size(p, 1) == 2)

  % Write nodes
  fprintf(fp,'  <mesh celltype="triangle" dim="2">\n');
  disp('Writing vertices...')
  fprintf(fp,'    <vertices size="%d">\n',np);
  for n=1:np
    fprintf(fp,'      <vertex index="%d" x="%f" y="%f" z="0.0"/>\n', ...
	    n - 1, p(1, n), p(2, n));
  end
  fprintf(fp,'    </vertices>\n');

  % Write cells
  disp('Writing cells...')
  fprintf(fp,'    <cells size="%d">\n',nt);
  for n=1:nt
    fprintf(fp,'      <triangle index="%d" v0="%d" v1="%d" v2="%d"/>\n', ...
	    n - 1, t(1, n) - 1, t(2, n) - 1, t(3, n) - 1);
  end
  fprintf(fp,'    </cells>\n');
  fprintf(fp,'  </mesh>\n');
  fprintf(fp,'</dolfin>\n');

% 3D mesh
elseif (size(p, 1) == 3)

  % Write nodes
  disp('Writing nodes...')
  fprintf(fp,'  <mesh>\n');
  fprintf(fp,'    <vertices size="%d">\n',np);
  for n=1:np
    fprintf(fp,'      <vertex name="%d" x="%f" y="%f" z="%f"/>\n', ...
            n - 1, p(1, n), p(2, n), p(3, n));
  end
  fprintf(fp,'    </vertices>\n');

  % Write cells
  disp('Writing cells...')
  fprintf(fp,'    <cells size="%d">\n',nt);
  for n=1:nt
    fprintf(fp,'      <tetrahedron name="%d" n0="%d" n1="%d" n2="%d" n3="%d"/>\n', ...
            n-1, t (1, n) - 1, t(2, n) - 1, t(3, n) - 1, t(4, n) - 1);
  end
  fprintf(fp,'    </cells>\n');
  fprintf(fp,'  </mesh>\n');
  fprintf(fp,'</dolfin>\n');

end

% Close file
fclose(fp);
disp('Done')
