% Load solution
poisson

% Compute exact solution
x = points(1,:);
y = points(2,:);
z = points(3,:);

uu = sin(pi*x) .* sin(2.0*pi*y) .* sin(3.0*pi*z);

% Compute error
e = u - uu;
E = max(abs(e));
disp(['|| e ||_oo = ' num2str(E)])
