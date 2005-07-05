function xmlmatrix(filename, A)

% XMLMATRIX - SAVE MATRIX TO DOLFIN IN XML FORMAT
%
% Usage: xmlmatrix(filename, A)
%
%   A - a matrix
%
% Copyright (C) 2004 Georgios Foufas.
% Licensed under the GNU GPL Version 2.
%
% Modified by Anders Logg 2004-2005.
%
% First added:  2004-02-10
% Last changed: 2005

% Tolerance for sparse matrix
tol = 1e-16;

% Open file
fp = fopen(filename,'w');

%Get matrix size
nrows = size(A,1);
ncols = size(A,2);

% Write header
fprintf(fp,'<?xml version="1.0" encoding="UTF-8"?>\n\n');
fprintf(fp,'<dolfin xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">\n');

% Write matrix values
disp('Writing matrix...')
fprintf(fp,'  <sparsematrix rows="%d" columns="%d">\n',nrows,ncols);  

for i=1:nrows
  
  % Compute size of row
  size = round(length(find(abs(A(i,:)) > tol)));
  
  fprintf(fp,'    <row row="%d" size="%d"/>\n', i-1, size);
  
  for j=1:ncols
    element = A(i,j);
    if abs(element) > tol
      fprintf(fp,'      <element column="%d" value="%f"/>\n', j-1, A(i,j));
    end
  end
  
  fprintf(fp,'    </row>\n');

end

fprintf(fp,'  </sparsematrix>\n');
fprintf(fp,'</dolfin>\n');  

% Close file
fclose(fp);
disp('Done')
