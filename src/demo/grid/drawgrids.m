% Copyright (C) 2003 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Draw grids from the 2D grid refinement example.

% Load the grids
disp('Loading grids')
grids2D

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
disp(['Found ' num2str(n) ' grids in ' num2str(levels) ' levels'])

% Draw grids
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

% Draw the finest grid
figure(2)
clf
pdemesh(points{n}, edges{n}, cells{n})
