function writeinp(filename,p,t)

% WRITEINP - SAVE MATLAB GRID TO INP FORMAT
%
% Usage: writeinp(filename,p,t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)

% Open file
fp = fopen(filename,'w');

np = size(p,2);
nt = size(t,2);

% Write header
fprintf(fp,'%d %d 0 0 0\n',np,nt);

% Write nodes
disp('Writing nodes...')
for i=1:np
	fprintf(fp,'%d %f %f 0.0\n',i,p(1,i),p(2,i));
end

% Write cells
disp('Writing cells...')
for i=1:nt
	fprintf(fp,'%d 0 tri %d %d %d\n',i,t(1,i),t(2,i),t(3,i));
end

% Close file
fclose(fp);
disp('Done')
