function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(length(alpha), num_iters+1);
for i = 1:length(alpha)
    J_history(1, i) = computeCostMulti(X, y, theta);
end

for i = 1:length(alpha)
    for iter = 2:num_iters+1
        difference_vector = X*theta - y;
        learning_term = difference_vector'*X;
        theta = theta - alpha(i)/m*learning_term';
    % ============================================================

    % Save the cost J in every iteration    
        J_history(iter,i) = computeCostMulti(X, y, theta);
    end
end
