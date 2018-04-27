% This function calculates the X feature matrix for use in a linear
% regression (optimal linear decoder) model. 
%
% Inputs are:
% - features = an array containing all the feature matrices for this
% subject; make sure that the number of rows of 'features' equals
% 'numChannels'
% - nTimeBins = the number of time bins preceding the current moment
% - numChannels = the number of channels for this subject
%
% Output is: X matrix

function x = CalcXMatrix(feat1, feat2, feat3, feat4, feat5, feat6, nTimeBins, numChannels)

    N = nTimeBins; % number of time bins before
    M = size(feat1,2)-N+1; % number of rows for X matrix
    v = numChannels; % number of channels for subject
    nF = 6; % number of features
    nC = v*N*nF+1; % number of columns
    x = []; % empty X matrix
    % Construct X matrix
    for i = 1:M

        currRow = [1];

        % Iterate through each channel
        for j = 1:v

            % Add 3 values from each feature matrix, at row j
            currRow = [currRow feat1(j,i:(i+2))];
            currRow = [currRow feat2(j,i:(i+2))];
            currRow = [currRow feat3(j,i:(i+2))];
            currRow = [currRow feat4(j,i:(i+2))];
            currRow = [currRow feat5(j,i:(i+2))];
            currRow = [currRow feat6(j,i:(i+2))];

        end

        % Add current row to matrix
        x = [x; currRow];

    end
    
end