function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
m = length(y); % number of training examples
hypothesis = sigmoid(X*theta);
sigmoid_matrix = [log(hypothesis), log(1 - hypothesis)];
y_vector = [-y'; y'-1];
grad = theta;
sum = 0;
for i = 1:m
    sum = sum + sigmoid_matrix(i,:)*y_vector(:,i);
end
J = sum/m;
for i = 1:size(theta)
    grad(i) = (X(:,i)'*(sigmoid(X*theta) - y))/m;
end
multiplied = (sigmoid_matrix .* y_vector') * ones(2,1);
new_J = (ones(1, size(X,1)) * multiplied)/m;
difference_rowvector = (sigmoid(X*theta) - y)';
grad_new = (difference_rowvector * X)'/m;
assert(new_J - J < 1e-5);
%assert(grad - grad_new)
end
