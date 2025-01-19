function W = findPCs(X, K)
% This function finds the first K principle components of X

% ====================== YOUR CODE HERE ======================
% Instructions: First, compute the estimated variance of X, denoted as S
%               Second, compute eigenvalues and corresponding 
%               eigenvectors of S
%               Third, sort the eigenvalues in a descend order
%               Four, find the top K principle components with the largest
%               eigenvalues
% Hints:        try functions cov(), eig() 
%
S = cov(X);
[eigenvectors, eigenvalues] = eig(S);
eigenvalues = diag(eigenvalues);
[~, order] = sort(eigenvalues, 'descend');
sort_eigenvectors = eigenvectors(:, order);
W = sort_eigenvectors(:, 1:K);


% ============================================================
end