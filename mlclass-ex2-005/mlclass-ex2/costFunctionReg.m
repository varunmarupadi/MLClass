function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
sigmoid_matrix = [log(sigmoid(X*theta)), log(1 - sigmoid(X*theta))];
y_vector = [-y'; y'-1];
sum = 0;
grad = theta;
for i = 1:m
    sum = sum + sigmoid_matrix(i,:)*y_vector(:,i);
end
theta_d = [0; theta(2:end)];
J = sum/m + (theta_d'*theta_d)*lambda/2/m;
for i = 2:size(theta)
    grad(i) = (X(:,i)'*(sigmoid(X*theta) - y))/m + lambda*theta_d(i)/m;
end

end
