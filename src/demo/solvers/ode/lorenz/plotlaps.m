% Copyright (C) 2005 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Plot the number of laps around each fixed point for Lorenz.

load lorenz.txt

t     = lorenz(:, 1);
n0    = lorenz(:, 3);
n1    = lorenz(:, 4);
alpha = lorenz(:, 5);

figure(1)
clf

subplot(2, 1, 1)
plot(t, n0)
xlabel('t')
ylabel('n1')

subplot(2, 1, 2)
plot(t, n1)
xlabel('t')
ylabel('n2')

figure(2)
clf

subplot(2, 1, 1)
plot(t(1:1339), alpha(1:1339))
xlabel('t')
ylabel('n1 / n2')

subplot(2, 1, 2)
plot(t, alpha)
xlabel('t')
ylabel('n1 / n2')

% Compute exponent for y = 1 - alpha = c*t^p
n = length(t);
x = t(100:n);
y = abs(1 - alpha(100:n));
p = polyfit(log(x), log(y), 1);

disp(['alpha at end of integration: ' num2str(alpha(length(alpha)))])
disp(['exponent: ' num2str(p(1))])
