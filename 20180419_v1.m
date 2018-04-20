%% Downloading data

% Download all data from the website

% Initiate session
% Open datasets for subject 1
session = IEEGSession('I521_Sub1_Training_ecog','pkshah','pks_ieeglogin.bin');
session.openDataSet('I521_Sub1_Training_dg');
session.openDataSet('I521_Sub1_Leaderboard_ecog');

% Open datasets for subject 2
session.openDataSet('I521_Sub2_Training_ecog');
session.openDataSet('I521_Sub2_Training_dg');
session.openDataSet('I521_Sub2_Leaderboard_ecog');

% Open datasets for subject 3
session.openDataSet('I521_Sub3_Training_ecog');
session.openDataSet('I521_Sub3_Training_dg');
session.openDataSet('I521_Sub3_Leaderboard_ecog');

% Download data for subject 1
ecogData1 = session.data(1).getvalues(1:300000,1:62);
dgData1 = session.data(2).getvalues(1:300000,1:5);
testECOG1 = session.data(3).getvalues(1:147500,1:62);

% Download data for subject 2
ecogData2 = session.data(4).getvalues(1:300000,1:48);
dgData2 = session.data(5).getvalues(1:300000,1:5);
testECOG2 = session.data(6).getvalues(1:147500,1:48);

% Download data for subject 3
ecogData3 = session.data(7).getvalues(1:300000,1:64);
dgData3 = session.data(8).getvalues(1:300000,1:5);
testECOG3 = session.data(9).getvalues(1:147500,1:64);

%% Downsampling

% Vars to hold downsampled data glove data for each subject
dsDG1 = zeros(6000,5);
dsDG2 = zeros(6000,5);
dsDG3 = zeros(6000,5);

for i = 1:5
    dsDG1(:,i) = decimate(dgData1(:,i), 50);
    dsDG2(:,i) = decimate(dgData2(:,i), 50);
    dsDG3(:,i) = decimate(dgData3(:,i), 50);
end
clear i;

%% Saving data

save('finalProjectData.mat','dgData1','dgData2','dgData3','dsDG1','dsDG2',...
    'dsDG3','ecogData1','ecogData2','ecogData3','testECOG1','testECOG2','testECOG3');

%% Calculate features for training data

% The six features we are using are the same as what are recommended in the
% assignment:
% 1) average time-domain voltage
% 2) average frequency-domain magnitude in the 5-15 Hz frequency band
% 3) average frequency-domain magnitude in the 20-25 Hz frequency band
% 4) average frequency-domain magnitude in the 75-115 Hz frequency band
% 5) average frequency-domain magnitude in the 125-160 Hz frequency band
% 6) average frequency-domain magnitude in the 160-175 Hz frequency band

sr = 1000; % 1000 Hz
winLen = 100/1e3; % window length is 100ms
winDisp = 50/1e3; % window displacement is 50ms

% FEATURE 1
% Average time-domain voltage
AvgVoltage = @(signal) mean(signal);

% Generate features for each 100ms window, separated by 50ms, in each
sub1Feature1 = [];
for i = 1:62
   
    % Use MovingWinFeats to calculate AvgVoltage for each channel for each
    % Note that we had to amend MovingWinFeats in order to make this work
    % properly for the dataset (removed floor() function)
    result1 = MovingWinFeats(ecogData1(:,i),sr,winLen,winDisp,AvgVoltage);
    
    % Add results to feature matrix for each subject
    sub1Feature1 = [sub1Feature1; result1];
    
end

sub2Feature1 = [];
for i = 1:48
    
    result2 = MovingWinFeats(ecogData2(:,i),sr,winLen,winDisp,AvgVoltage);
    sub2Feature1 = [sub2Feature1; result2];
    
end

sub3Feature1 = [];
for i = 1:64
    
    result3 = MovingWinFeats(ecogData3(:,i),sr,winLen,winDisp,AvgVoltage);
    sub3Feature1 = [sub3Feature1; result3];
    
end

% FEATURES 2-6
% Average frequency-domain magnitude in all frequency bands
windowSamples = 100; % 100 samples in 100ms
overlapSamples = 50; % 50 samples in overlap

% Calculated for subject 1
sub1Feature2 = [];
sub1Feature3 = [];
sub1Feature4 = [];
sub1Feature5 = [];
sub1Feature6 = [];
for i = 1:62
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result1 = mean(abs(spectrogram(ecogData1(:,i), 100, 50, [5/1e3 15/1e3])));
    sub1Feature2 = [sub1Feature2; result1];
    result1 = mean(abs(spectrogram(ecogData1(:,i), 100, 50, [20/1e3 25/1e3])));
    sub1Feature3 = [sub1Feature3; result1];
    result1 = mean(abs(spectrogram(ecogData1(:,i), 100, 50, [75/1e3 115/1e3])));
    sub1Feature4 = [sub1Feature4; result1];
    result1 = mean(abs(spectrogram(ecogData1(:,i), 100, 50, [125/1e3 160/1e3])));
    sub1Feature5 = [sub1Feature5; result1];
    result1 = mean(abs(spectrogram(ecogData1(:,i), 100, 50, [160/1e3 175/1e3])));
    sub1Feature6 = [sub1Feature6; result1];
    
end

% Calculated for subject 2
sub2Feature2 = [];
sub2Feature3 = [];
sub2Feature4 = [];
sub2Feature5 = [];
sub2Feature6 = [];
for i = 1:48
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result2 = mean(abs(spectrogram(ecogData2(:,i), 100, 50, [5/1e3 15/1e3])));
    sub2Feature2 = [sub2Feature2; result2];
    result2 = mean(abs(spectrogram(ecogData2(:,i), 100, 50, [20/1e3 25/1e3])));
    sub2Feature3 = [sub2Feature3; result2];
    result2 = mean(abs(spectrogram(ecogData2(:,i), 100, 50, [75/1e3 115/1e3])));
    sub2Feature4 = [sub2Feature4; result2];
    result2 = mean(abs(spectrogram(ecogData2(:,i), 100, 50, [125/1e3 160/1e3])));
    sub2Feature5 = [sub2Feature5; result2];
    result2 = mean(abs(spectrogram(ecogData2(:,i), 100, 50, [160/1e3 175/1e3])));
    sub2Feature6 = [sub2Feature6; result2];
    
end

% Calculated for subject 3
sub3Feature2 = [];
sub3Feature3 = [];
sub3Feature4 = [];
sub3Feature5 = [];
sub3Feature6 = [];
for i = 1:64
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result3 = mean(abs(spectrogram(ecogData3(:,i), 100, 50, [5/1e3 15/1e3])));
    sub3Feature2 = [sub3Feature2; result3];
    result3 = mean(abs(spectrogram(ecogData3(:,i), 100, 50, [20/1e3 25/1e3])));
    sub3Feature3 = [sub3Feature3; result3];
    result3 = mean(abs(spectrogram(ecogData3(:,i), 100, 50, [75/1e3 115/1e3])));
    sub3Feature4 = [sub3Feature4; result3];
    result3 = mean(abs(spectrogram(ecogData3(:,i), 100, 50, [125/1e3 160/1e3])));
    sub3Feature5 = [sub3Feature5; result3];
    result3 = mean(abs(spectrogram(ecogData3(:,i), 100, 50, [160/1e3 175/1e3])));
    sub3Feature6 = [sub3Feature6; result3];
    
end

%% Save features

save('finalProjectFeatures.mat','sub1Feature1','sub1Feature2','sub1Feature3',...
    'sub1Feature4','sub1Feature5','sub1Feature6','sub2Feature1','sub2Feature2',...
    'sub2Feature3','sub2Feature4','sub2Feature5','sub2Feature6','sub3Feature1',...
    'sub3Feature2','sub3Feature3','sub3Feature4','sub3Feature5','sub3Feature6');

%% Compute X (R) matrix for training linear prediction (Subject 1)

% Add value to beginning of feature matrix to make it have 1 dimension with
% 6000 values
sub1Feature1B = [mean(sub1Feature1')' sub1Feature1];
sub1Feature2B = [mean(sub1Feature2')' sub1Feature2];
sub1Feature3B = [mean(sub1Feature3')' sub1Feature3];
sub1Feature4B = [mean(sub1Feature4')' sub1Feature4];
sub1Feature5B = [mean(sub1Feature5')' sub1Feature5];
sub1Feature6B = [mean(sub1Feature6')' sub1Feature6];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub1Feature1B,2)-N+1; % number of rows for "normal" X matrix
v = 62; % number of channels for subject 1
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
trainX1 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub1Feature1B(j,i:(i+2))];
        currRow = [currRow sub1Feature2B(j,i:(i+2))];
        currRow = [currRow sub1Feature3B(j,i:(i+2))];
        currRow = [currRow sub1Feature4B(j,i:(i+2))];
        currRow = [currRow sub1Feature5B(j,i:(i+2))];
        currRow = [currRow sub1Feature6B(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    trainX1 = [trainX1; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependX1 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependX1 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependX1 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub1Feature1B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature2B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature3B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature4B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature5B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature6B(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependX1 = [prependX1; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependX1 and trainX1 to form trainX1 matrix with same number of rows as
% original training data
trainX1 = [prependX1; trainX1];

%% Compute X (R) matrix for training linear prediction (Subject 2)

% Add value to beginning of feature matrix to make it have 1 dimension with
% 6000 values
sub2Feature1B = [mean(sub2Feature1')' sub2Feature1];
sub2Feature2B = [mean(sub2Feature2')' sub2Feature2];
sub2Feature3B = [mean(sub2Feature3')' sub2Feature3];
sub2Feature4B = [mean(sub2Feature4')' sub2Feature4];
sub2Feature5B = [mean(sub2Feature5')' sub2Feature5];
sub2Feature6B = [mean(sub2Feature6')' sub2Feature6];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub2Feature1B,2)-N+1; % number of rows for "normal" X matrix
v = 48; % number of channels for subject 2
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
trainX2 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub2Feature1B(j,i:(i+2))];
        currRow = [currRow sub2Feature2B(j,i:(i+2))];
        currRow = [currRow sub2Feature3B(j,i:(i+2))];
        currRow = [currRow sub2Feature4B(j,i:(i+2))];
        currRow = [currRow sub2Feature5B(j,i:(i+2))];
        currRow = [currRow sub2Feature6B(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    trainX2 = [trainX2; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependX2 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependX2 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependX2 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub2Feature1B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature2B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature3B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature4B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature5B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature6B(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependX2 = [prependX2; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependX2 and trainX2 to form trainX2 matrix with same number of rows as
% original training data
trainX2 = [prependX2; trainX2];

%% Compute X (R) matrix for training linear prediction (Subject 3)

% Add value to beginning of feature matrix to make it have 1 dimension with
% 6000 values
sub3Feature1B = [mean(sub3Feature1')' sub3Feature1];
sub3Feature2B = [mean(sub3Feature2')' sub3Feature2];
sub3Feature3B = [mean(sub3Feature3')' sub3Feature3];
sub3Feature4B = [mean(sub3Feature4')' sub3Feature4];
sub3Feature5B = [mean(sub3Feature5')' sub3Feature5];
sub3Feature6B = [mean(sub3Feature6')' sub3Feature6];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub3Feature1B,2)-N+1; % number of rows for "normal" X matrix
v = 64; % number of channels for subject 3
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
trainX3 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub3Feature1B(j,i:(i+2))];
        currRow = [currRow sub3Feature2B(j,i:(i+2))];
        currRow = [currRow sub3Feature3B(j,i:(i+2))];
        currRow = [currRow sub3Feature4B(j,i:(i+2))];
        currRow = [currRow sub3Feature5B(j,i:(i+2))];
        currRow = [currRow sub3Feature6B(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    trainX3 = [trainX3; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependX3 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependX3 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependX3 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub3Feature1B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature2B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature3B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature4B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature5B(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature6B(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependX3 = [prependX3; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependX3 and trainX3 to form trainX3 matrix with same number of rows as
% original training data
trainX3 = [prependX3; trainX3];


%% Cross validation (Subject 1)

% Starting with subject 1 data, set apart 10% of the data as a testing set
% for later
permuted = randperm(size(trainX1,1)); % permute indices of X randomly
lim = floor(size(trainX1,1)/10); % find limit for 10% of the data
testingIdx = permuted(1:lim); % indices for test set
trainingIdx = permuted((lim+1):length(permuted)); % indices for training set
testingX1 = trainX1(testingIdx,:); % testing data
trainingX1 = trainX1(trainingIdx,:); % training data
testingDG1 = dsDG1(testingIdx,:); % testing response variables
trainingDG1 = dsDG1(trainingIdx,:); % training response variables

% Compute filter matrices
beta1 = mldivide((trainingX1'*trainingX1),(trainingX1'*trainingDG1));

% Compute predictions
yhat1 = testingX1*beta1;

% Compute correlation between predicted values and actual values
finger1Corr = corr(yhat1(:,1),testingDG1(:,1));
finger2Corr = corr(yhat1(:,2),testingDG1(:,2));
finger3Corr = corr(yhat1(:,3),testingDG1(:,3));
finger4Corr = corr(yhat1(:,4),testingDG1(:,4));
finger5Corr = corr(yhat1(:,5),testingDG1(:,5));
averageCorr = mean([finger1Corr finger2Corr finger3Corr finger5Corr]);

%% Cross validation (Subject 2)

% Starting with subject 2 data, set apart 10% of the data as a testing set
% for later
permuted = randperm(size(trainX2,1)); % permute indices of X randomly
lim = floor(size(trainX2,1)/10); % find limit for 10% of the data
testingIdx = permuted(1:lim); % indices for test set
trainingIdx = permuted((lim+1):length(permuted)); % indices for training set
testingX2 = trainX2(testingIdx,:); % testing data
trainingX2 = trainX2(trainingIdx,:); % training data
testingDG2 = dsDG2(testingIdx,:); % testing response variables
trainingDG2 = dsDG2(trainingIdx,:); % training response variables

% Compute filter matrices
beta2 = mldivide((trainingX2'*trainingX2),(trainingX2'*trainingDG2));

% Compute predictions
yhat2 = testingX2*beta2;

% Compute correlation between predicted values and actual values
finger1Corr = corr(yhat2(:,1),testingDG2(:,1));
finger2Corr = corr(yhat2(:,2),testingDG2(:,2));
finger3Corr = corr(yhat2(:,3),testingDG2(:,3));
finger4Corr = corr(yhat2(:,4),testingDG2(:,4));
finger5Corr = corr(yhat2(:,5),testingDG2(:,5));
averageCorr = mean([finger1Corr finger2Corr finger3Corr finger5Corr]);

%% Cross validation (Subject 3)

% Starting with subject 3 data, set apart 10% of the data as a testing set
% for later
permuted = randperm(size(trainX3,1)); % permute indices of X randomly
lim = floor(size(trainX3,1)/10); % find limit for 10% of the data
testingIdx = permuted(1:lim); % indices for test set
trainingIdx = permuted((lim+1):length(permuted)); % indices for training set
testingX3 = trainX3(testingIdx,:); % testing data
trainingX3 = trainX3(trainingIdx,:); % training data
testingDG3 = dsDG3(testingIdx,:); % testing response variables
trainingDG3 = dsDG3(trainingIdx,:); % training response variables

% Compute filter matrices
beta3 = mldivide((trainingX3'*trainingX3),(trainingX3'*trainingDG3));

% Compute predictions
yhat3 = testingX3*beta3;

% Compute correlation between predicted values and actual values
finger1Corr = corr(yhat3(:,1),testingDG3(:,1));
finger2Corr = corr(yhat3(:,2),testingDG3(:,2));
finger3Corr = corr(yhat3(:,3),testingDG3(:,3));
finger4Corr = corr(yhat3(:,4),testingDG3(:,4));
finger5Corr = corr(yhat3(:,5),testingDG3(:,5));
averageCorr = mean([finger1Corr finger2Corr finger3Corr finger5Corr]);

%% Compute features for testing data

sr = 1000; % 1000 Hz
winLen = 100/1e3; % window length is 100ms
winDisp = 50/1e3; % window displacement is 50ms

% FEATURE 1
% Average time-domain voltage
AvgVoltage = @(signal) mean(signal);

% Generate features for each 100ms window, separated by 50ms, in each
sub1Feature1T = [];
for i = 1:62
   
    % Use MovingWinFeats to calculate AvgVoltage for each channel for each
    % Note that we had to amend MovingWinFeats in order to make this work
    % properly for the dataset (removed floor() function)
    result1 = MovingWinFeats(testECOG1(:,i),sr,winLen,winDisp,AvgVoltage);
    
    % Add results to feature matrix for each subject
    sub1Feature1T = [sub1Feature1T; result1];
    
end

sub2Feature1T = [];
for i = 1:48
    
    result2 = MovingWinFeats(testECOG2(:,i),sr,winLen,winDisp,AvgVoltage);
    sub2Feature1T = [sub2Feature1T; result2];
    
end

sub3Feature1T = [];
for i = 1:64
    
    result3 = MovingWinFeats(testECOG3(:,i),sr,winLen,winDisp,AvgVoltage);
    sub3Feature1T = [sub3Feature1T; result3];
    
end

% FEATURES 2-6
% Average frequency-domain magnitude in all frequency bands
windowSamples = 100; % 100 samples in 100ms
overlapSamples = 50; % 50 samples in overlap

% Calculated for subject 1
sub1Feature2T = [];
sub1Feature3T = [];
sub1Feature4T = [];
sub1Feature5T = [];
sub1Feature6T = [];
for i = 1:62
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result1 = mean(abs(spectrogram(testECOG1(:,i), 100, 50, [5/1e3 15/1e3])));
    sub1Feature2T = [sub1Feature2T; result1];
    result1 = mean(abs(spectrogram(testECOG1(:,i), 100, 50, [20/1e3 25/1e3])));
    sub1Feature3T = [sub1Feature3T; result1];
    result1 = mean(abs(spectrogram(testECOG1(:,i), 100, 50, [75/1e3 115/1e3])));
    sub1Feature4T = [sub1Feature4T; result1];
    result1 = mean(abs(spectrogram(testECOG1(:,i), 100, 50, [125/1e3 160/1e3])));
    sub1Feature5T = [sub1Feature5T; result1];
    result1 = mean(abs(spectrogram(testECOG1(:,i), 100, 50, [160/1e3 175/1e3])));
    sub1Feature6T = [sub1Feature6T; result1];
    
end

% Calculated for subject 2
sub2Feature2T = [];
sub2Feature3T = [];
sub2Feature4T = [];
sub2Feature5T = [];
sub2Feature6T = [];
for i = 1:48
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result2 = mean(abs(spectrogram(testECOG2(:,i), 100, 50, [5/1e3 15/1e3])));
    sub2Feature2T = [sub2Feature2T; result2];
    result2 = mean(abs(spectrogram(testECOG2(:,i), 100, 50, [20/1e3 25/1e3])));
    sub2Feature3T = [sub2Feature3T; result2];
    result2 = mean(abs(spectrogram(testECOG2(:,i), 100, 50, [75/1e3 115/1e3])));
    sub2Feature4T = [sub2Feature4T; result2];
    result2 = mean(abs(spectrogram(testECOG2(:,i), 100, 50, [125/1e3 160/1e3])));
    sub2Feature5T = [sub2Feature5T; result2];
    result2 = mean(abs(spectrogram(testECOG2(:,i), 100, 50, [160/1e3 175/1e3])));
    sub2Feature6T = [sub2Feature6T; result2];
    
end

% Calculated for subject 3
sub3Feature2T = [];
sub3Feature3T = [];
sub3Feature4T = [];
sub3Feature5T = [];
sub3Feature6T = [];
for i = 1:64
    
    % Compute spectral features for this channel for each frequency band
    % and add to result to feature matrix
    result3 = mean(abs(spectrogram(testECOG3(:,i), 100, 50, [5/1e3 15/1e3])));
    sub3Feature2T = [sub3Feature2T; result3];
    result3 = mean(abs(spectrogram(testECOG3(:,i), 100, 50, [20/1e3 25/1e3])));
    sub3Feature3T = [sub3Feature3T; result3];
    result3 = mean(abs(spectrogram(testECOG3(:,i), 100, 50, [75/1e3 115/1e3])));
    sub3Feature4T = [sub3Feature4T; result3];
    result3 = mean(abs(spectrogram(testECOG3(:,i), 100, 50, [125/1e3 160/1e3])));
    sub3Feature5T = [sub3Feature5T; result3];
    result3 = mean(abs(spectrogram(testECOG3(:,i), 100, 50, [160/1e3 175/1e3])));
    sub3Feature6T = [sub3Feature6T; result3];
    
end

%% Compute X matrix for testing data (Subject 1)

% Add value to beginning of feature matrix to make it have 1 additional
% column
sub1Feature1TB = [mean(sub1Feature1T')' sub1Feature1T];
sub1Feature2TB = [mean(sub1Feature2T')' sub1Feature2T];
sub1Feature3TB = [mean(sub1Feature3T')' sub1Feature3T];
sub1Feature4TB = [mean(sub1Feature4T')' sub1Feature4T];
sub1Feature5TB = [mean(sub1Feature5T')' sub1Feature5T];
sub1Feature6TB = [mean(sub1Feature6T')' sub1Feature6T];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub1Feature1TB,2)-N+1; % number of rows for "normal" X matrix
v = 62; % number of channels for subject 1
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
testX1 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub1Feature1TB(j,i:(i+2))];
        currRow = [currRow sub1Feature2TB(j,i:(i+2))];
        currRow = [currRow sub1Feature3TB(j,i:(i+2))];
        currRow = [currRow sub1Feature4TB(j,i:(i+2))];
        currRow = [currRow sub1Feature5TB(j,i:(i+2))];
        currRow = [currRow sub1Feature6TB(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    testX1 = [testX1; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependXT1 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependXT1 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependXT1 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub1Feature1TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature2TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature3TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature4TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature5TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub1Feature6TB(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependXT1 = [prependXT1; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependXT1 and testX1 to form testX1 matrix with same number of rows as
% original training data
testX1 = [prependXT1; testX1];

%% Compute X matrix for testing data (Subject 2)

% Add value to beginning of feature matrix to make it have 1 additional
% column
sub2Feature1TB = [mean(sub2Feature1T')' sub2Feature1T];
sub2Feature2TB = [mean(sub2Feature2T')' sub2Feature2T];
sub2Feature3TB = [mean(sub2Feature3T')' sub2Feature3T];
sub2Feature4TB = [mean(sub2Feature4T')' sub2Feature4T];
sub2Feature5TB = [mean(sub2Feature5T')' sub2Feature5T];
sub2Feature6TB = [mean(sub2Feature6T')' sub2Feature6T];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub2Feature1TB,2)-N+1; % number of rows for "normal" X matrix
v = 48; % number of channels for subject 1
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
testX2 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub2Feature1TB(j,i:(i+2))];
        currRow = [currRow sub2Feature2TB(j,i:(i+2))];
        currRow = [currRow sub2Feature3TB(j,i:(i+2))];
        currRow = [currRow sub2Feature4TB(j,i:(i+2))];
        currRow = [currRow sub2Feature5TB(j,i:(i+2))];
        currRow = [currRow sub2Feature6TB(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    testX2 = [testX2; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependXT2 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependXT2 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependXT2 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub2Feature1TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature2TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature3TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature4TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature5TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub2Feature6TB(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependXT2 = [prependXT2; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependXT2 and testX2 to form testX2 matrix with same number of rows as
% original training data
testX2 = [prependXT2; testX2];

%% Compute X matrix for testing data (Subject 3)

% Add value to beginning of feature matrix to make it have 1 additional
% column
sub3Feature1TB = [mean(sub3Feature1T')' sub3Feature1T];
sub3Feature2TB = [mean(sub3Feature2T')' sub3Feature2T];
sub3Feature3TB = [mean(sub3Feature3T')' sub3Feature3T];
sub3Feature4TB = [mean(sub3Feature4T')' sub3Feature4T];
sub3Feature5TB = [mean(sub3Feature5T')' sub3Feature5T];
sub3Feature6TB = [mean(sub3Feature6T')' sub3Feature6T];

% Construct one part of the X matrix
N = 3; % number of time bins before
M = size(sub3Feature1TB,2)-N+1; % number of rows for "normal" X matrix
v = 64; % number of channels for subject 1
nF = 6; % number of features
nC = v*N*nF+1; % number of columns
testX3 = []; % empty X matrix
% Iterate through first rows to construct "normal" X matrix
for i = 1:M
    
    currRow = [1];
    
    % Iterate through each channel
    for j = 1:v
        
        % Add 3 values from each feature matrix, at row j
        currRow = [currRow sub3Feature1TB(j,i:(i+2))];
        currRow = [currRow sub3Feature2TB(j,i:(i+2))];
        currRow = [currRow sub3Feature3TB(j,i:(i+2))];
        currRow = [currRow sub3Feature4TB(j,i:(i+2))];
        currRow = [currRow sub3Feature5TB(j,i:(i+2))];
        currRow = [currRow sub3Feature6TB(j,i:(i+2))];
        
    end
    
    % Add current row to matrix
    testX3 = [testX3; currRow];
        
end

% Create matrix for (N-1) rows to prepend to existing X matrix
prependXT3 = []; % empty matrix
pad = 0; % value to pad matrix with where we don't have data
padCount = N-1; % counter to help us populate prependXT3 matrix
for i = 1:N-1 % iterate through all N-1 rows of prependXT3 matrix
    
    currRow = [1];
    
    for j = 1:v % iterate through all channels
    
        currRow = [currRow zeros(1,padCount) sub3Feature1TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature2TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature3TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature4TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature5TB(j,1:i)];
        currRow = [currRow zeros(1,padCount) sub3Feature6TB(j,1:i)];
        
    end
    
    % Add current row to matrix
    prependXT3 = [prependXT3; currRow];
    
    padCount = padCount - 1;
    
end

% Combine prependXT3 and testX3 to form testX3 matrix with same number of rows as
% original training data
testX3 = [prependXT3; testX3];

%% Computing filter matrices and predictions (for all 3 subjects)

% Subject 1
beta1T = mldivide((trainX1'*trainX1),(trainX1'*dsDG1)); % train coeffs on training data
yhat1T = testX1*beta1T; % predict using testing data

% Subject 2
beta2T = mldivide((trainX2'*trainX2),(trainX2'*dsDG2)); % train coeffs on training data
yhat2T = testX2*beta2T; % predict using testing data

% Subject 3
beta3T = mldivide((trainX3'*trainX3),(trainX3'*dsDG3)); % train coeffs on training data
yhat3T = testX3*beta3T; % predict using testing data

%% Interpolation to produce final output (for all 3 subjects)

% Expand from 2950 rows per subject's predictions to 147500 rows per
% subject's predictions

% Subject 1
finalY1Finger1 = spline(1:2950, yhat1T(:,1),0:1/50:(2950-1/50));
finalY1Finger2 = spline(1:2950, yhat1T(:,2),0:1/50:(2950-1/50));
finalY1Finger3 = spline(1:2950, yhat1T(:,3),0:1/50:(2950-1/50));
finalY1Finger4 = spline(1:2950, yhat1T(:,4),0:1/50:(2950-1/50));
finalY1Finger5 = spline(1:2950, yhat1T(:,5),0:1/50:(2950-1/50));

% Subject 2
finalY2Finger1 = spline(1:2950, yhat2T(:,1),0:1/50:(2950-1/50));
finalY2Finger2 = spline(1:2950, yhat2T(:,2),0:1/50:(2950-1/50));
finalY2Finger3 = spline(1:2950, yhat2T(:,3),0:1/50:(2950-1/50));
finalY2Finger4 = spline(1:2950, yhat2T(:,4),0:1/50:(2950-1/50));
finalY2Finger5 = spline(1:2950, yhat2T(:,5),0:1/50:(2950-1/50));

% Subject 3
finalY3Finger1 = spline(1:2950, yhat3T(:,1),0:1/50:(2950-1/50));
finalY3Finger2 = spline(1:2950, yhat3T(:,2),0:1/50:(2950-1/50));
finalY3Finger3 = spline(1:2950, yhat3T(:,3),0:1/50:(2950-1/50));
finalY3Finger4 = spline(1:2950, yhat3T(:,4),0:1/50:(2950-1/50));
finalY3Finger5 = spline(1:2950, yhat3T(:,5),0:1/50:(2950-1/50));

% Combine output into one data structure called predicted_dg
predicted_dg = {};
predicted_dg{1} = [finalY1Finger1' finalY1Finger2' finalY1Finger3' finalY1Finger4' finalY1Finger5'];
predicted_dg{2} = [finalY2Finger1' finalY2Finger2' finalY2Finger3' finalY2Finger4' finalY2Finger5'];
predicted_dg{3} = [finalY3Finger1' finalY3Finger2' finalY3Finger3' finalY3Finger4' finalY3Finger5'];
predicted_dg = predicted_dg';

% Save predicted_dg to .mat file for submission
save('readyPlayerOne_predictions.mat','predicted_dg');