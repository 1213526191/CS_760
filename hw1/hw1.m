mu = [0, 0];
sigma = [1, 0; 0, 1];
x = mvnrnd ( mu, sigma, 100 );
figure(1)
scatter(x(:,1), x(:,2))


mu = [1, -1];
sigma = [1, 0; 0, 1];
x = mvnrnd ( mu, sigma, 100 );
figure(2)
scatter(x(:,1), x(:,2))


mu = [0, 0];
sigma = [2, 0; 0, 2];
x = mvnrnd ( mu, sigma, 100 );
figure(3)
scatter(x(:,1), x(:,2))

mu = [0, 0];
sigma = [2, 0.2; 0.2, 2];
x = mvnrnd ( mu, sigma, 100 );
figure(4)
scatter(x(:,1), x(:,2))

mu = [0, 0];
sigma = [2, -0.2; -0.2, 2];
x = mvnrnd ( mu, sigma, 100 );
figure(5)
scatter(x(:,1), x(:,2))