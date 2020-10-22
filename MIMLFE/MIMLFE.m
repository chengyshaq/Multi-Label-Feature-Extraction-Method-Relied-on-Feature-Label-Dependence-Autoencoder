function [P] = MIMLFE(X, Y, parameter)

C = parameter.C;
para = parameter.para;

[N1, d] = size(X);
[N2, q] = size(Y);
if(N1 ~= N2) disp('The number of training instances in X and Y is not equal'); end
N =min(N1, N2);

%_____________________________________
n = size(Y,1);
Omega_train = RBF(X, para);
% Omega_test = RBF(X,para,Xt);
OutputWeight=((Omega_train+speye(n)/C)\([X,Y]));

KX=(Omega_train * OutputWeight);
KX = KX(1:d,:);
A = KX*KX';

H=eye(N,N)-1/N;

XHY = X'*H*Y;
B = XHY*XHY';
beta = parameter.beta;

norm_fro = norm(A-A', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix A= ', num2str(norm_fro)));
end

norm_fro = norm(B-B', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix B = ', num2str(norm_fro)));
end

beta00 = beta;

if(beta00<0.0)beta00=0.0;end
if(beta00>1.0)beta00=1.0;end
disp(strcat('Beta value = ', num2str(beta00)));

G = (1-beta00)*A + beta00*B;

norm_fro = norm(G-G', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix = ', num2str(norm_fro)));
end

if(parameter.rank==1)
    rankG = rank(G);
    disp(strcat('The rank of matrix G = ',num2str(rankG)));
else
    rankG = N;
end

[V, D] = eig(G);

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, rankG, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);
