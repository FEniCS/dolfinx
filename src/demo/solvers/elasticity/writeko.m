function h = writeko(p, e, t, u)

for i = 1:size(u, 2)
  fname = sprintf("mesh%.5d.xml", i)
  pde2ko(p, e, t, '', u{i}', fname);
end
