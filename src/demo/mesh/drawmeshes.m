% Copyright (C) 2003 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% First added:  2003-10-21
% Last changed: 2003

% Load the meshes
disp('Loading meshes')
meshes2D

% Count the number of levels
n = length(points);
levels = 0;
count = 0;
for i=1:n
  count = count + i;
  if ( count >= n )
     levels = i;
     break;
  end
end
disp(['Found ' num2str(n) ' meshes in ' num2str(levels) ' levels'])

% Draw meshes
figure(1)
clf
count = 1;
for i=1:levels
  for j=1:i
    subplot(levels, levels, (i-1)*levels + j)
    pdemesh(points{count}, edges{count}, cells{count})
    axis([0 1 0 1])
    axis off
    count = count + 1;
  end
end

% Draw the finest mesh
figure(2)
clf
pdemesh(points{n}, edges{n}, cells{n})
