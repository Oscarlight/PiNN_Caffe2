N = 1000;
res = zeros(1, N);
vds = linspace(0, 0.2, N);
for i = 1:N
    res(i) = wrappedModel(0.2, 0.0, vds(i), 1);
end

plot(vds, res, 'LineWidth', 1.5);