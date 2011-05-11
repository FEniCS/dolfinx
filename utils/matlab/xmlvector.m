function xmlvector(filename, x)

% XMLVECTOR - SAVE VECTOR TO DOLFIN IN XML FORMAT
%
% Usage: xmlvector(filename, x)
%
%   x - a vector
%
% Copyright (C) 2004 Georgios Foufas
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
%
% First added:  2004-02-10
% Last changed: 2005


% Open file
fp = fopen(filename,'w');

% Row or column  vector??
ssize = size(x,1);
if(ssize == 1)
  ssize = size(x,2);
end

% Write header
fprintf(fp,'<?xml version="1.0" encoding="UTF-8"?>\n\n');
fprintf(fp,'<dolfin xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">\n');

% Write vector values
disp('Writing vector...')
fprintf(fp,'  <vector size="%d">\n',ssize);  

for n = 1:ssize
  fprintf(fp,'    <element row="%d" value="%f"/>\n', n-1, x(n));
end

fprintf(fp,'  </vector>\n');
fprintf(fp,'</dolfin>\n');  
 
% Close file
fclose(fp);
disp('Done')
