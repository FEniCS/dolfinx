function writexml(filename,p,t)
                                                                                                                                          
% WRITEXML - SAVE MATLAB 2D (AND FEMLAB 3D) GRID TO DOLFIN XML FORMAT
%
% Usage: writexml(filename,p,t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)
%
% Copyright (C) 2003 Erik Svensson.
% Licensed under the GNU GPL Version 2.

% Open file
fp = fopen(filename,'w');

np = size(p,2);
nt = size(t,2);

% Write header
fprintf(fp,'<?xml version="1.0" encoding="UTF-8"?>\n\n');
fprintf(fp,'<dolfin
xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">\n');

% Write nodes
disp('Writing nodes...')
fprintf(fp,'  <grid>\n');
fprintf(fp,'    <nodes size="%d">\n',np);

% 2D grid
if (size(p,1) == 2)
  
  for n=1:np
    fprintf(fp,'      <node name="%d" x="%f" y="%f" z="0.0"/>\n', ...
	    n-1, p(1,n), p(2,n));
  end
  
  fprintf(fp,'    </nodes>\n');
  fprintf(fp,'    <cells size="%d">\n',nt);
  
  % Write cells
  % disp('Writing cells...')
  for n=1:nt
    fprintf(fp,'      <triangle name="%d" n0="%d" n1="%d" n2="d"/>\n', \
	    ...
	    n-1,t(1,n)-1,t(2,n)-1,t(3,n)-1);
  end
  
% 3D grid
elseif (size(p,1) == 3)

  for n=1:np
    fprintf(fp,'      <node name="%d" x="%f" y="%f" z="%f"/>\n', ...
            n-1,p(1,n),p(2,n),p(3,n));
  end
  
  fprintf(fp,'    </nodes>\n');
  fprintf(fp,'    <cells size="%d">\n',nt);
  
  % Write cells
  disp('Writing cells...')
  for n=1:nt
    fprintf(fp,'      <tetrahedron name="%d" n0="%d" n1="%d" n2="%d" n3="%d"/>\n', ...
            n-1,t(1,n)-1,t(2,n)-1,t(3,n)-1,t(4,n)-1);
  end

end
                                                                                                                                          
fprintf(fp,'    </cells>\n');
fprintf(fp,'  </grid>\n');
fprintf(fp,'</dolfin>');
                                                                                                                                          
% Close file
fclose(fp);
disp('Done')
