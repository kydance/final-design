clc, clear
close all

DATA_PREFIX = '/home/tianen/doc/MachineLearningData/htd/';
DATA = 'sandiego_plane';

%% Load the hyperspectral image and ground truth
DATA_SUFFFIX = '.mat';
DATASET = strcat(DATA_PREFIX, DATA, DATA_SUFFFIX);
DATA_POST = strcat(DATA_PREFIX, DATA, "_post", DATA_SUFFFIX);
load(DATASET)
[w, h, bs] = size(data);

index = find(map==1);
temp = reshape(data, [w*h, bs]);
target = temp(index, :);
d = mean(target, 1);

result_coarse = CEM(data, d');
%% Binarization
k = 0.15;
result_binary = result_coarse;
[a, b] = find(result_binary > k);
for i = 1:size(b)
  result_binary(a(i),b(i)) = 1;
end
[c, d] = find(result_binary <1);
for i=1:size(d)
  result_binary(c(i),d(i)) = 0;
end

%% Get training data
data_r = hyperConvert2d(data);
data_r = data_r';
ratio = 0.75;
iMax = size(c, 1);
% all background samples
for i = 1:iMax
  background(i,:) = data(c(i), d(i), :);
end
rowrank = randperm(size(background, 1));
background = background(rowrank, :);
m = ceil(ratio*iMax);
train_data = background(1:m,:);
val_data = background(m+1:iMax,:);
%save ('./train_data','train_data');
%save ('./val_data','val_data');

%% Figures
figure,
subplot(1, 3, 1); imagesc(map), axis image, axis off; title('ground truth')
subplot(1, 3, 2); imagesc(result_coarse), axis image, axis off; title('coarse detection')
subplot(1, 3, 3); imagesc(result_binary), axis image, axis off; title('binarization')

%% Save data
%save ('./result_coarse','result_coarse');
%save ('./result_binary','result_binary');

ori_data = data;
save(DATA_POST, 'ori_data', 'train_data', 'map', 'd','result_coarse')
