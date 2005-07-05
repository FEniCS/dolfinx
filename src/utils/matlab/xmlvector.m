function xmlvector(filename, x)

% XMLVECTOR - SAVE VECTOR TO DOLFIN IN XML FORMAT
%
% Usage: xmlvector(filename, x)
%
%   x - a vector
%
% Copyright (C) 2004 Georgios Foufas.
% Licensed under the GNU GPL Version 2.
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
