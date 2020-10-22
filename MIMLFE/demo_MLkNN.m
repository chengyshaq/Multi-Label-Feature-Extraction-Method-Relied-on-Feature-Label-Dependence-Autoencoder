clear;clc
addpath('Measures');
addpath('data');
addpath('Algorithms');

load society.mat;
 
% Set ML-kNN para
Num = 10;
Smooth = 1;
tic;
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
endtime = toc