% Plot solution using MATLAB

% Load solution
stokes

% Plot solution with pdeplot
pdeplot(points, edges, cells, 'flowdata', u(:,1:2))
axis([0 1 0 1])
xlabel('x')
ylabel('y')
title('Stokes')
