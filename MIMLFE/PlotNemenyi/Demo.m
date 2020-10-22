% 数据按照这个模式，自己写入数据
% results：
%           行表示：实验数据集
%           列表示：实验算法
% 例如这里的第一行共有6列表示：在数据集Yeast gen中各算法的排位。第一列到第六列分别表示算法：
% 'ML-RKELM','ML-ASRKELM','RELM','ML-ELM-RBF','ML-RBF','MLKNN'
% 代码nemenyi为论文：An extensive experimental comparison of methods for multi-label learning
% Madjarov G, Kocev D, Gjorgjevikj D, et al. An extensive experimental comparison of methods for multi-label learning[J]. Pattern Recognition, 2012, 45(9):3084-3104.
% load HL.mat;

% 写自己需要对比的算法
Names = {'OS','MDDMp','MVMD','PCA','MLSI','wMLDA','MIMLFE'}; % Algorithm Names

% 写自己电脑的任何目录,'./'表示当前目录
OutputFolder  = './';                     % Output folder

% 写各评价指标：HL OE CV RL AP et.cl
Outname = 'AP';                                                % Output name
drawNemenyi(results, Names, OutputFolder, Outname);

