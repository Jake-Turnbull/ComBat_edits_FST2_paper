%% Check consitency between the different ComBat methods:


%% Define out data 

% Data equation = y_ijg = a_j + X_ij B_ij + gamma_jg + delta_jg epsilon_ijg


subject_num = 2000;
feature_num = 100;

% Consider data to be Z-scores, -5 to 5 in value (maximum) and normally
% distributed in each feature
data = randn(subject_num,feature_num);

% Define a normally distributed age effect and random negative beta
% coefficients

min_age = 45;
max_age = 75;
age_distribution = (max_age-min_age).*rand(subject_num,1) + min_age;
beta_age = (rand(1,feature_num))-0.5;


% Define random indices and split 50% male 50% female as well as define
% beta cofficient
sex = ones(subject_num,1);
ismale = randperm(subject_num,subject_num/2);
sex(ismale,:) = 2;
beta_sex = ((1.2-1.05).*rand(1,feature_num) + 1.05);

%% Define the batch effect:

% Assume the feature mean is a random permutation around zero (+ or - 0.2)
feature_mean = 0.1.*randn(feature_num,1);


batch = ones(subject_num,1);

% Define 2 batches
batch_2_indices = randperm(subject_num,round(subject_num/2));
batch(batch_2_indices,1) = 2;


% Define additive component of batch 
% Normal distribution for additive in keeping with ComBat logic
gamma = normrnd(0,2,[1,100]);
% Gamma distribution for multiplicative in keeping with ComBat logic
delta = random('Gamma',1.2,0.4,[1,100]);

%% Construct data


% Add the covariate effects to the data
Covariates = [age_distribution, sex];
Covariates_demeaned = Covariates-mean(Covariates);
Betas = [beta_age; beta_sex];
cov_effects = (Covariates_demeaned * Betas);

% Add the multiplicative terms
% Multiply random subject effects by delta 
data_1 = data;
data_1(batch==2,:) = data_1(batch==2,:) .* delta;

% add randome effects 
data_1(batch==2,:) = data_1(batch==2,:) + gamma;

% Construct the equation

Y = feature_mean' + cov_effects + data_1;

%% Test that there is a batch effect

addpath('/Users/jacob.turnbull/Documents/MATLAB/Functions/')

d_vals = Cohens_d(Y(batch==1,:),Y(batch==2,:));

plot(d_vals,'.');
hold on
plot(d_vals)

%% Plot the ratio of variance

var_batch1 = var(Y(batch==1,:));
var_batch2 = var(Y(batch==2,:));

var_ratio = var_batch1./var_batch2;
plot(var_ratio);

%% Test harmonization methods:
addpath('ComBat-M/')

disp('Testing with no covariates')
[Y_1,~]= combat(Y',batch,[],1);

[Y_2,~,~,~,~,~,~,~] = combat_modified(Y',batch,[],1);

tf = isequal(Y_1,Y_2);
if tf == 1
    disp('The two arrays are exactly matched, Combat functionality preserved')
else
    error('The two arrays do not match!!! Review script to find bug')
end


%% Test when covariates are included
disp('Testing with covariates')
addpath('ComBat-M/')

disp('Testing with covariates')
[Y_1,~]= combat(Y',batch,Covariates,1);

[Y_2,~,~,~,~,~,~,~] = combat_modified(Y',batch,Covariates,1);

tf = isequal(Y_1,Y_2);
if tf == 1
    disp('The two arrays are exactly matched, Combat functionality preserved')
else
    error('The two arrays do not match!!! Review script to find bug')
end

%% Testing with covariates without demeaning 

disp('Testing with covariates')

disp('Testing with covariates without demeaning first, check not demeaned against demeaned')
[Y_1,~]= combat_modified(Y',batch,Covariates,1);

[Y_2,~,~,~,~,~,~,~] = combat_modified(Y',batch,Covariates_demeaned,1);

tf = isequal(Y_1,Y_2);

if tf == 1
    disp('The two arrays are exactly matched, Combat functionality preserved')
else
    warning(['The two arrays do not match!!! This is the part where we test the' ...
        ' difference between applying the raw covariates against demeaned'])
end

%% Testing demeaning internally vs prior

disp('Testing demeaning prior vs internally')

disp('Testing with covariates without demeaning first, check not demeaned against demeaned')
[Y_1]= combat_modified_test(Y',batch,Covariates_demeaned,1,'RegressCovariates',true);

[Y_2,~,~,~,~,~,~,~] = combat_modified(Y',batch,Covariates,1,'RegressCovariates',true);

tf = isequal(Y_1,Y_2);

if tf == 1
    disp('The two arrays are exactly matched, Combat functionality preserved')
else
    warning(['The two arrays do not match!!! This is the part where we test the' ...
        ' difference between applying the raw covariates against demeaned'])
end

%%