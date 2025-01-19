%% SDSU Machine Learning Course (CompE510/EE600/CompE596)
%% Programming Assignment:  PCA
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
%  PCA assignment. 
%
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., number of 
%  principle components K).


% Initialization
clear ; close all; clc

%% =========== Part 1: Load Image ============================

fprintf('===== Part 1: Load Image ===== \n');
% Instructions: The following code loads and display image bluebird.jped
%               Hint: you can use functions "imread()" and "imshow()"
% ============================================================


% ====================== YOUR CODE HERE ======================
% Step 1: read the image and store it in "img"
img = imread('bluebird.jpeg')

% Step 2: display the image
imshow(img)

% Step 3: scale the pixel values so that they fall within [0,1]
img = double(img)/255;
% ============================================================


fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Perform PCA to Compress the Image =============
fprintf('===== Part 2: Perform PCA to Compress the Image ===== \n');
% Instructions: The following code perform PCA on each color channel to 
%               reduce feature dimensions. Tune the value of K to see its impact.  
%               

% ====================== YOUR CODE HERE ======================
% Step 1: Configure the number of principle components 
K = 30  % you need to change this
% ============================================================


% find the number of RGB pixels in the image
[n,m,c] = size(img);
img_compressed = zeros(n,m,c);
for rgb = 1:3
    % Center the data 
    X = img(:,:,rgb) - mean(img(:,:,rgb));
    
% ====================== YOUR CODE HERE ======================
    W = findPCs(X, K); % W stores the principle components
    Xt = PCAtransform(X,W); % Transform X to a lower dimension using PCA
    X_inv = PCAtransform_inv(Xt,W);  % recover the original image using compressed data
    X_reconstruct = X_inv + mean(img(:,:,rgb));
    X_reconstruct = max(min(X_reconstruct, 1), 0);
    img_compressed(:,:,rgb) = X_reconstruct % you need to change this. 
% ============================================================
end

img_compressed = uint8(img_compressed*255);
% store the compressed image 
imwrite(img_compressed, 'bluebird_compressed.jpeg');
% display the compressed image
fprintf('Compression is done. \n');
figure;
imshow(img_compressed)