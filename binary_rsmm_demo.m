clear;
clc;
dbstop if error;

addpath('./libqp/matlab');
%%====================load data ========================
load data.mat
% X: input training data, size of dim1 x dim2 x #training sample
% y: corresponding label for training data, size of #training sample x 1 {1,-1}
% X_test: input testing data, size of dim1 x dim2 x #testing sample
% y_test: corresponding label for testing data, size of #testing sample x 1
% {1,-1}

%% ==================parameter setting ==================
c = 1;   % lambda_1 in paper
tau = 0.1;  % lambda_2 in paper
gamma = 0.01; % lambda_3 in paper


fprintf('lambda1=%f,lambda2=%f,lambda3=%f\n',c,tau,gamma);
max_iter = 50;
sz = size(X);
sz_test = size(X_test);
% Train the model with training data
[W,b,L,S,obj] = SparLR_ADMM(X,y,c,tau,gamma,max_iter);
w = reshape(W,[sz(1)*sz(2),1]);
l = reshape(L,[sz(1)*sz(2),sz(3)]);
y_hat = sign(l'*w+b);
% Calculate training accuracy
acc_train = length(find(y == y_hat))/sz(3);
fprintf('the training accuracy is %f \n',acc_train);
% Matrix decomposition
[L_test,S_test] = rpca(X_test,tau,gamma,max_iter);
% Calculate the testing accuracy
l_test = reshape(L_test,[sz_test(1)*sz_test(2),sz_test(3)]);
y_test_hat = sign(l_test'*w+b);
acc_test = length(find(y_test == y_test_hat))/sz_test(3);
fprintf('the testing accuracy is %f \n\n', acc_test);

