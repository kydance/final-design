clc, clear
close all

DATA_PREFIX = '/home/tianen/doc/MachineLearningData/htd/';
DATA = 'aerorit';

DATA_RESULT = '/home/tianen/doc/_XiDian/___FinalDesign/code/final-design/experiments/aerorit/aae_aerorit/aae_abs_2023.12.29-17:03:58-847/aerorit.mat';

%% Load the hyperspectral image and ground truth
DATA_SUFFFIX = '.mat';
DATASET = strcat(DATA_PREFIX, DATA, DATA_SUFFFIX);
load(DATASET)


%% Load the hyperspectral image and ground truth
load(DATA_RESULT)
load(DATASET)

tic
ori_data = ori_data(:,:,1:50);
data=double(ori_data);
[w, h, bs] = size(data);
data = hyperNormalize(data);
data_r = hyperConvert2d(data)';
% reconstruct_result = hyperNormalize(reconstruct_result);
reconstruct_result = hyperNormalize(y);
reconstruct_result = hyperConvert2d(reconstruct_result)';

%% Parameters setup
lamda = 10;
% FOR AeroRIT dataset,max=200; FOR XiongAn dataset,max=4
max = 4;

%% Difference
for i = 1: w*h
  sam(i)= hyperSam(data_r(i,:), reconstruct_result(i,:));
end
sam = reshape(sam , w, h);
SAM = hyperNormalize( sam );
%% Binary the difference
% output  = nonlinear(SAM, lamda, max );% FOR AeroRIT dataset
output  = nonlinear(result_coarse, lamda, max );% FOR XiongAn dataset
B = SAM.* output;
res=B;

toc
[FPR,TPR,thre] = myPlot3DROC( map, B);
auc = -trapz(FPR,TPR);
fpr = -trapz(FPR,thre);
figure, imagesc(B), axis image, axis off
