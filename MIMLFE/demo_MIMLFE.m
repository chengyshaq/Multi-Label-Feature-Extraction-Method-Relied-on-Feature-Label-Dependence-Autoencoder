clear;clc
addpath('Measures');
addpath('data');
addpath('Validation');

load Arts.mat;
 
% Set ML-kNN para
Num = 10;
Smooth = 1;

%----------------------
Y = train_target';
Y(Y==-1) = 0;

% Set para
parameter.beta = 0;
parameter.rank = 1;

% parameter.para = 1000;
% parameter.C = 1000;

dim = size(train_data,2);
for i = 1:1
ratio = ceil(i*0.1*dim);
% ratio = 1.999;
[PX,TPX] = exc_MIMLFE(train_data,Y,test_data,ratio);
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