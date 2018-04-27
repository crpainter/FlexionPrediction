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

function x = CalcXMatrixLiang(band1Feat,band2Feat,band3Feat,nTimeBins,numChannels)

    N = nTimeBins; % number of time bins before
    M = size(band1Feat,2)-N+1; % number of rows for X matrix
    v = numChannels; % number of channels for subject
    nF = 3; % number of features
    nC = v*N*nF+1; % number of columns
    x = []; % empty X matrix
    % Construct X matrix
    for i = 1:M

        currRow = [1];

        % Iterate through each channel
        for j = 1:v

            % Add N values from each feature matrix, at row j
            currRow = [currRow band1Feat(j,i:(i+2))];
            currRow = [currRow band2Feat(j,i:(i+2))];
            currRow = [currRow band3Feat(j,i:(i+2))];

        end

        % Add current row to matrix
        x = [x; currRow];

    end
    
end