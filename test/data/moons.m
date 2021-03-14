data = csvread("./moons.csv")(2:end, :);

scatter(data(:, 1), data(:, 2), [], data(:, 3), "filled");

fit = csvread("./moons_fit.csv");
step = fit(2, 2) - fit(1, 2);
fx = min(fit(:, 1)):step:max(fit(:, 1));
fy = min(fit(:, 2)):step:max(fit(:, 2));
[xx, yy] = meshgrid(fx, fy);
zz = reshape(cast(fit(:, 3) > 0, "double"), 60, 60);

hold on
c = contourf(xx, yy, zz);
colormap(jet);
scatter(data(:, 1), data(:, 2), 7, "filled");
xlim([-1.5, 2.5]);
ylim([-1, 1.5]);

print("./moons.png");
