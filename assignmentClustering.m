%% SDSU Machine Learning Course (CompE510/EE600/CompE596)
%% Programming Assignment:  K-Means Clustering 
%
%  Instructions
%  ------------
% 
%  Dataset comes from: 
%   https://leafyplace.com/types-of-birds/
% 
%
% 
%  This file contains code that helps you get started on the
%  k-means clustering assignment. 
%
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  the number of clusters).


% Initialization
clear ; close all; clc

%% =========== Part 1: Load Image ============================

fprintf('===== Part 1: Load Image ===== \n');

% Instructions: The following code loads and display image bluebird.jped
%               Hint: you can use functions "imread()" and "imshow()"
% ============================================================

% ====================== YOUR CODE HERE ======================
% Step 1: read the image and store it in "img"
img = imread('bluebird.jpeg') %you need to change this

% Step 2: display the image
imshow(img)

% ============================================================

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Cluster Pixels with K-means =============
fprintf('===== Part 2: Cluster Pixels with K-means ===== \n');

% Instructions: The following code cluster the data points (pixels) into K 
%               clusters. Tune the value of K to see its impact.  
%
%               Procedure: 
%               First initilize the centroids
%               Then iterate following steps until convergence (iterate for a certain number of iterations)
%                   a: Assign each pixel into the cluster with centroid closest to it
%                   b: Update the centroids 
% ============================================================


% ====================== YOUR CODE HERE ======================
% Step 1: Configure the number of clusters
K = 20  % you need to change this

% Step 2: scale the pixel values so that they fall within [0,1]
img = double(img)/255;

% Step 3: convert the image as a 2d matrix 
% Hint: if an image is of dimension (n,m,c), then reshape it 
%       to a matrix of dimension (n*m,c)
c = size(img, 3);
img_reshape = reshape(img, [], size(img, c))

% Step 4: initilize the cluster centroids
centroids = rand(K,c)

% Step 5: iterate to cluster the pixels
num_iterations = 100 % change this
SSE = zeros(1, num_iterations);
for iter = 1:num_iterations
    distances = pdist2(img_reshape, centroids, 'euclidean');  % All pairwise distances
    
    % Assign each pixel to the closest centroid
    [~, clusterAssignments] = min(distances, [], 2);  % Find the closest centroid for each pixel
    
    % Step 5b: Update centroids
    newCentroids = zeros(K, 3);  % Initialize new centroids matrix
    
    for i = 1:K
        % Calculate the mean of all pixels assigned to cluster i
        clusterPixels = img_reshape(clusterAssignments == i, :);  % Get pixels assigned to cluster i
        if ~isempty(clusterPixels)
            newCentroids(i, :) = mean(clusterPixels, 1);  % Update centroid to the mean of the assigned pixels
        end
    end
    
    % Update the centroids
    centroids = newCentroids;
    sse = 0;  % Initialize SSE for this iteration
    numPixels = size(img_reshape, 1);
    for i = 1:numPixels
        % Sum the squared distance between each pixel and its assigned centroid
        sse = sse + sum((img_reshape(i, :) - centroids(clusterAssignments(i), :)).^2);
    end
    
    % Store the SSE for this iteration
    SSE(iter) = sse;
end

% ============================================================
% Step 6: Plot a graph that shows the change of SSE (Sum of Squared Error) as a way
% to check convergence
figure;
plot(1:num_iterations, SSE, '-o');
xlabel('Iteration');
ylabel('Sum of Squared Errors (SSE)');
title('Change in SSE over K-means Iterations');


fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Construct the Compressed Image ================
fprintf('===== Part 3: Construct the Compressed Image ===== \n');

% Instructions: The following code construct and display the compressed image.
%               Hint: pixels within the same cluster have the same value as
%               the cluster centroid.  
%
% 
% ====================== YOUR CODE HERE ======================
img_compressed = centroids(clusterAssignments, :);
img_compressed = reshape(img_compressed, size(img));
img_compressed = uint8(img_compressed * 255);
% ============================================================

% store the compressed image 
imwrite(img_compressed, 'bluebird_compressed.jpeg');
% display the compressed image
fprintf('Compression is done. \n');
figure;
imshow(img_compressed)


