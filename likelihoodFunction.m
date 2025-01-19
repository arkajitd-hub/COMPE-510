function [l, grad] = likelihoodFunction(beta, X, y)
%LIKELIHOODFUNCTION Computes log likelihood using beta as the parameter 
%   for logistic regression and the gradient of the log likelihood function
%   w.r.t. to the parameters.

% You need to return the following variables correctly 
l = 0; 
grad = zeros(size(beta)); 
m = length(y);

% ====================== YOUR CODE HERE =====================
% Instructions: Compute the log-likelihood (l) of a particular choice of beta
%               averaged over all training examples, i.e., l = 1/num_train * (...) 
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the log-likelihood function w.r.t. each 
%               parameter in beta
%
% Note: grad should have the same dimensions as beta
%

   % Compute the gradient
prediction = sigmoid(X*beta);
epsilon = 1e-10;
prediction = min(max(prediction, epsilon), 1 - epsilon);

l = 1/m * sum(y .* log(prediction) + (1 - y) .* log(1 - prediction));
grad = 1/m * (X' * (y - prediction));

% =============================================================

end
