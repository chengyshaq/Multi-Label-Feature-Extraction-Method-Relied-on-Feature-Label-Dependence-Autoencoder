clear;clc
addpath('Measures');
addpath('data');
addpath('Validation');

load Arts.mat;
 
%----------------------
Y = train_target';
% Y(Y==-1) = 0;

% Set para
% parameter.beta = 1;
% parameter.rank = 1;
% parameter.C = 1;
% parameter.para = 1;

d = size(train_data,2);

[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = HKELM(train_data,Y,test_data,test_target);
% for i = 1:1
% % parameter.ratio = ceil(0.02*i*d);
% parameter.ratio = d;
% [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = HKELM(train_data,Y,test_data,test_target,parameter);
%     HL(i)=HammingLoss;
%     RL(i)=RankingLoss;
%     OE(i)=OneError;
%     CV(i)=Coverage;
%     AP(i)=Average_Precision;
% end

