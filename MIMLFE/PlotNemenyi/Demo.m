% ���ݰ������ģʽ���Լ�д������
% results��
%           �б�ʾ��ʵ�����ݼ�
%           �б�ʾ��ʵ���㷨
% ��������ĵ�һ�й���6�б�ʾ�������ݼ�Yeast gen�и��㷨����λ����һ�е������зֱ��ʾ�㷨��
% 'ML-RKELM','ML-ASRKELM','RELM','ML-ELM-RBF','ML-RBF','MLKNN'
% ����nemenyiΪ���ģ�An extensive experimental comparison of methods for multi-label learning
% Madjarov G, Kocev D, Gjorgjevikj D, et al. An extensive experimental comparison of methods for multi-label learning[J]. Pattern Recognition, 2012, 45(9):3084-3104.
% load HL.mat;

% д�Լ���Ҫ�Աȵ��㷨
Names = {'OS','MDDMp','MVMD','PCA','MLSI','wMLDA','MIMLFE'}; % Algorithm Names

% д�Լ����Ե��κ�Ŀ¼,'./'��ʾ��ǰĿ¼
OutputFolder  = './';                     % Output folder

% д������ָ�꣺HL OE CV RL AP et.cl
Outname = 'AP';                                                % Output name
drawNemenyi(results, Names, OutputFolder, Outname);

