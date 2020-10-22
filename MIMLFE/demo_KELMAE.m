clear;clc
addpath('Measures');
addpath('data');
addpath('Algorithms');

load Arts.mat;
 
% Set ML-kNN para
Num = 10;
Smooth = 1;

%----------------------
Y = train_target';
Y(Y==-1) = 0;

% Set para
parameter.beta = 0.5;
parameter.rank = 1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
parameter.para = 1;
parameter.C = 1;
para = 1;
C = 1;
dim = size(train_data,2);
for i = 1:10
parameter.ratio = ceil(0.1*i*dim);
% parameter.ratio = 1.999;

% P= LASSOAE(train_data, Y, parameter);
P= MIMLFE(train_data, Y, parameter);
PX = train_data*P;
PTX = test_data*P;

% % KELMAE
% n1 = size(train_data,1);
% Omega_train = RBF(train_data, para);
% KX=((Omega_train+speye(n1)/C)\PX);
% KX = KX(:,1:d);
% n2 = size(test_data,1);
% Omega_test = RBF(test_data,para);
% KTX=((Omega_test+speye(n2)/C)\TPX);
% KTX = KTX(:,1:d);

% Runing 
[Prior,PriorN,Cond,CondN]=MLKNN_train(PX,train_target,Num,Smooth);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MLKNN_test(PX,train_target,PTX,test_target,Num,Prior,PriorN,Cond,CondN);
    HL(i)=HammingLoss;
    RL(i)=RankingLoss;
    OE(i)=OneError;
    CV(i)=Coverage;
    AP(i)=Average_Precision;
end