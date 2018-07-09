clear;
clc;
dbstop if error;

addpath('./libqp/matlab');
%%====================load data ========================
% load multiclassdata.mat
% X: input training data, size of dim1 x dim2 x #training sample
% y: corresponding label for training data, size of #training sample x 1,
% {1,2,3,4...}
% X_test: input testing data, size of dim1 x dim2 x #testing sample
% y_test: corresponding label for testing data, size of #testing sample x
% 1{1,2,3,4,...}


%% ==================parameter setting ==================
c = 1;   % lambda_1 in paper
tau = 0.1;  % lambda_2 in paper
gamma = 0.01; % lambda_3 in paper
fprintf('lambda1=%f,lambda2=%f,lambda3=%f\n',c,tau,gamma);


numClass = length(unique(y));
max_iter = 10;
sz = size(X);
sz_test = size(X_test);
numDim = sz(1)*sz(2);
W = zeros(numDim,numClass);
b = zeros(1,numClass);
for k = 1:numClass                
    y_new = (y == k) * 2 - 1;
    [W_tmp,b_tmp,L,S,obj] = SparLR_ADMM(X,y_new,c,tau,gamma,max_iter);
    b(1,k) = b_tmp;
    W(:,k) = reshape(W_tmp,[numDim,1]);
    fprintf('the %d the classifier is trained\n',k);
end


% calculate the testing accuracy
[L_test,S_test] = rpca(X_test,tau,gamma,max_iter);
l_test = reshape(L_test,sz_test(1)*sz_test(2),sz_test(3));

probValue_test = l_test'*W+repmat(b,[sz_test(3),1]);
probValue_test = probValue_test';
[~,ind_test] = max(probValue_test);
y_test_hat = ind_test';

acc_test = nnz(y_test_hat == y_test)/sz_test(3);
fprintf('the testing error is %.4f\n',1-acc_test);

