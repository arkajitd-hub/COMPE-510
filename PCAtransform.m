function Xt = PCAtransform(X,W)
% This function performs PCA to reduce the dimension of X
%   PCAtransform(X,W) transforms X to the K-dimensional PC space defined
%   by W; Each column of W is a principle component. 

% ====================== YOUR CODE HERE ======================

X_centre = X - mean(X);

% Transform X to the K-dimensional PC space defined by W
Xt = X_centre * W;

% ============================================================



end