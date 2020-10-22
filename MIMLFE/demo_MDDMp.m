clear;clc
addpath('Measures');
addpath('data');
addpath('Validation');

load arts.mat;
 
% Set ML-kNN para
Num = 10;
Smooth = 1;

%----------------------
Y = train_target';
Y(Y==-1) = 0;

% Set para
parameter.beta = 0.5;
parameter.rank = 1;

dim = size(train_data,2);
for i = 1:1

% parameter.ratio = ceil(0.1*i*dim);
parameter.ratio = 1.999;
[P] = MDDM_transform(train_data,Y, parameter, 0);
PX = train_data*P;
TPX = test_data*P;

% Runing 
[Prior,PriorN,Cond,CondN]=MLKNN_train(PX,train_target,Num,Smooth);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MLKNN_test(PX,train_target,TPX,test_target,Num,Prior,PriorN,Cond,CondN);
    HL(i)=HammingLoss;
    RL(i)=RankingLoss;
    OE(i)=OneError;
    CV(i)=Coverage;
    AP(i)=Average_Precision;
end