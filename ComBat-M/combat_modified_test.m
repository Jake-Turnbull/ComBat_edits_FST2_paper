function [bayesdata] = combat_modified_test(dat, batch, mod, parametric, NameValueArgs)
    
    % COMBAT_FULL_FUNCTIONALITY: Extended ComBat with optional reference batch
    % and empirical Bayes toggle, covariate regression and shrinkage
    %
    % INPUTS:
    %   dat               - [features x samples] data matrix
    %   batch             - [samples x 1] batch labels (numeric or categorical)
    %   mod               - [samples x covariates] design matrix for covariates
    %   parametric        - true = parametric adjustment; false = non-parametric
    
    % Options:
    %
    %   DeltaCorrection: call as DeltaCorrection = True/False
    %   Defaults to True if no argument is passed
    %
    %   UseEB: call as UseEB = True/False
    %   Defaults to True if no argument is passed
    %
    %   ReferenceBatch: Call as ReferenceBatch = int
    %   Defaults to no refrence batch if not given
    %
    %   UseGAMs: call as UseGAMS = True/False
    %   Defaults to not using GAM if argument not passed
    %
    %
    %
    % OUTPUTS:
    %   bayesdata        - harmonized data
    %   delta_star       - the delta correction for each batch
    %   gamma_star       - the gamma correction for each batch
    %
   
    arguments
    
        dat 
        batch 
        mod 
        parametric 
        NameValueArgs.DeltaCorrection = true
        NameValueArgs.UseEB = true
        NameValueArgs.ReferenceBatch = []
        NameValueArgs.UseGAMs = false
        NameValueArgs.RegressCovariates = false
        NameValueArgs.GammaCorrection = true
    end
    
    DeltaCorrection = NameValueArgs.DeltaCorrection;
    UseEB = NameValueArgs.UseEB;
    ReferenceBatch = NameValueArgs.ReferenceBatch;
    UseGAMs = NameValueArgs.UseGAMs;
    RegressCovariates = NameValueArgs.RegressCovariates;
    GammaCorrection = NameValueArgs.GammaCorrection;
    
    % Check reference batch
    if isempty(ReferenceBatch) == true
        disp('Reference batch not given, defaulting to no reference')
    elseif isempty(ReferenceBatch) ==  false
            ReferenceBatch = ReferenceBatch
            disp(' fitting prior estimates using this batch and leaving batch unchanged')
    end
    % Check is EB is to be used
    if UseEB == false
        disp('Empirical Bayes set to false, using first estimates from raw mean and variances')
    else
        disp('Empirical Bayes set to true')
    end
    
    % Placeholder for eventual GAMs implementation
    if UseGAMs == true
        disp('GAMs set to true, using a generalised additive model on continuous covariates')
        disp('GAMs not yet implemented into code, this call serves as helper function')
    end
    
    % Check if covariate effects are to be added back in
    if RegressCovariates == true
        disp('Regress Covariates set to true, skipping re-addition of OLS covariate estimates ')
    end
    
    % Check if the batch variance scaling is to be applied
    if DeltaCorrection == false
        disp('Delta correction set to False, applying no delta (scale) correction on data')
    end
    
    % Check if the batch mean shift correction is to be applied 
    if GammaCorrection == false
        disp('Gamma correction set to Flase, applying no gamma (scale) correction on data')
    end
    
    
    %%% ---------------- Begin ComBat code ----------------%%%
    
    % Compute the standard deviation across samples for each row (feature/gene)
    
    [sds] = std(dat')';
    
    % Find rows with zero standard deviation (i.e., constant values across samples)
    wh = find(sds == 0);
    [ns, ~] = size(wh);
    
    % If any rows are constant, throw an error
    if ns > 0
        error('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')
    end
    
    % Convert batch vector to categorical and then create dummy variables
    batchmod = categorical(batch);
    batchmod = dummyvar({batchmod});
    
    % Number of batches
    n_batch = size(batchmod, 2);
    levels = unique(batch);
    fprintf('[combat] Found %d batches\n', n_batch);
    
    % Create a cell array where each cell contains indices of samples in each batch
    batches = cell(0);
    for i = 1:n_batch
        batches{i} = find(batch == levels(i));
    end
    
    % Compute size of each batch and total number of samples
    n_batches = cellfun(@length, batches);
    n_array = sum(n_batches);
    
    % Construct design matrix including batch and additional covariates (mod)
    design = [batchmod mod];
    
    % Remove intercept column if present
    intercept = ones(1, n_array)';
    wh = cellfun(@(x) isequal(x, intercept), num2cell(design, 1));
    bad = find(wh == 1);
    design(:, bad) = [];
    
    fprintf('[combat] Adjusting for %d covariate(s) of covariate level(s)\n', size(design, 2) - size(batchmod, 2))
    
    % Check for confounding between batch and covariates
    if rank(design) < size(design, 2)
        nn = size(design, 2);
        if nn == (n_batch + 1)
            error('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        end
        if nn > (n_batch + 1)
            temp = design(:, (n_batch + 1):nn);
            if rank(temp) < size(temp, 2)
                error('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else
                error('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')
            end
        end
    end
    
    fprintf('[combat] Standardizing Data across features\n')
    
    % Estimate coefficients (B_hat) for the linear model using least squares
    B_hat = inv(design' * design) * design' * dat';   

    % Check for reference batch when taking covariate effects so that these
    % are taken only from reference batch
    if ~isempty(ReferenceBatch)
        % Unadjusted reference-batch mean from B_hat:
        ref_idx = find(levels == ReferenceBatch); 
        ref_samples = batches{ref_idx};  % sample indices of reference batch
        ref_batch_effect = B_hat(ref_idx, :);        % [1 x p] vector of feature means

        % If you have covariates, add their effect back in
        if ~isempty(design)
            tmp = design;
            tmp(:, 1:n_batch) = 0;
            Cov_effects = (tmp * B_hat)';
        end
        
        % Pooled variance in reference batch (feature-wise):
        
        % Build a design matrix just for the reference batch
        design_ref = design(ref_samples, :);
        
        % Compute predicted values using the full model
        predicted_ref = (design_ref * B_hat)';   % [features x n_ref_samples]
        
        % Compute residuals for reference batch only
        residuals_ref = dat(:, ref_samples) - predicted_ref;  % [features x n_ref_samples]
        
        % Compute variance across reference samples (like original ComBat, but ref-only)
        var_ref = mean(residuals_ref.^2, 2);  % [features x 1]
        
        disp(['The sice of the var_ref array is',size(var_ref)])

        % replicate ref mean & variance across all samples
        stand_mean = repmat(ref_batch_effect', 1, n_array);
        stand_mean = stand_mean + Cov_effects;
        
        var_pooled = var_ref;
        disp('The size of the var_pooled array is')
        size(var_pooled)

    else
        [n_features, n_samples] = size(dat);
        n_batch = size(batchmod, 2);
        n_batches = sum(batchmod, 1);  % Samples per batch
        n_array = sum(n_batches);
    
        fprintf('[combat] Standardizing Data across features\n')
	    B_hat = inv(design'*design)*design'*dat';
	    %Standarization Model
	    grand_mean = (n_batches/n_array)*B_hat(1:n_batch,:);
	    var_pooled = ((dat-(design*B_hat)').^2)*repmat(1/n_array,n_array,1);
	    stand_mean = grand_mean'*repmat(1,1,n_array);
	    % Making sure pooled variances are not zero:
	    wh = find(var_pooled==0);
	    var_pooled_notzero = var_pooled;
	    var_pooled_notzero(wh) = [];
	    var_pooled(wh) = median(var_pooled_notzero);

           
	    if not(isempty(design))
		    tmp = design;
		    tmp(:,1:n_batch) = 0;
		    stand_mean = stand_mean+(tmp*B_hat)';
        end	
    
    end

        % Optional: regress covariates
        X_cov = design(:, n_batch+1:end);
        %X_cov = X_cov-mean(X_cov,1);
        B_cov = B_hat(n_batch+1:end, :);
        Cov_effects = (X_cov * B_cov)';

   
    % Standardize the data
    s_data = (dat - stand_mean) ./ (sqrt(var_pooled)*repmat(1,1,n_array));
    
    % Estimate batch effect parameters using least squares
    fprintf('[combat] Fitting L/S model and finding priors\n')
    batch_design = design(:, 1:n_batch);
    gamma_hat = inv(batch_design' * batch_design) * batch_design' * s_data';
    disp(['Size of gamma hat: ', num2str(size(gamma_hat))])
    
    % Estimate batch-specific variances
    delta_hat = [];
    for i = 1:n_batch
        indices = batches{i};
        delta_hat = [delta_hat; var(s_data(:, indices)')];
    end
    disp(['Size of delta hat: ', num2str(size(delta_hat))])
    
    % Check if reference batch is given here:
    if isempty(ReferenceBatch) == true
        disp('Reference batch not given, defaulting to no reference')
    elseif isempty(ReferenceBatch) ==  false
            ReferenceBatch = ReferenceBatch
            disp(' fitting prior estimates using this batch and leaving batch unchanged')
    end
    
  % Compute hyperparameters: 
    if ~isempty(ReferenceBatch)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Reference

        % Compute the mean and variance of the standardized data for each feature
        disp('size of gamma hat is')
        size(gamma_hat)
        gamma_bar = mean(gamma_hat');  % mean for each feature (row vector)
        t2 = var(gamma_hat');       % variance for each feature (row vector)
        % Prevent zero variances
        t2(t2 == 0) = 1e-6;
       % variance (may be zero if only one batch)
        if t2 == 0
            t2 = 1e-6; % small regularization to avoid division by zero
        end
        % Compute hyperparameters a_prior, b_prior only 

        disp('Size of delta hat is')
        size(delta_hat)
        ref_delta = num2cell(delta_hat,2);
        
        a_prior=[]; b_prior=[];
        for i = 1:n_batch
            a_prior = [a_prior aprior(ref_delta{i})];
            b_prior = [b_prior bprior(ref_delta{i})];
        end

    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% No reference
        gamma_bar = mean(gamma_hat');
        
        t2 = var(gamma_hat');
        
        % Estimate hyperparameters (a, b) for inverse gamma prior
        delta_hat_cell = num2cell(delta_hat, 2);

        a_prior=[]; b_prior=[];
        for i = 1:n_batch
            a_prior = [a_prior aprior(delta_hat_cell{i})];
            b_prior = [b_prior bprior(delta_hat_cell{i})];
        end
    end
    
    % Apply the empirical Bayes estimates and check if reference batch has been
    % given

if parametric
    fprintf('[combat] Finding parametric adjustments\n')
    gamma_star =[]; delta_star=[];
    for i = 1:n_batch
        indices = batches{i};
            if ~isempty(ReferenceBatch)
                % Use priors from reference batch for shrinkage
                temp = itSol(s_data(:, indices), gamma_hat(i, :), delta_hat(i, :), ...
                             gamma_bar(i), t2(i), a_prior(i), b_prior(i), 0.001);
                gamma_star(i, :) = temp(1, :);
                delta_star(i, :) = temp(2, :);
                % Reference batch: no correction
                if i == ref_idx
                        gamma_star(ref_idx, :) = zeros(1, size(dat, 1)); % no shift
                        delta_star(ref_idx, :) = ones(1, size(dat, 1));  % no scale correction
                end
            else
            temp = itSol(s_data(:,indices),gamma_hat(i,:),delta_hat(i,:), ...
                gamma_bar(i),t2(i),a_prior(i),b_prior(i), 0.001);
            gamma_star = [gamma_star; temp(1,:)];
            delta_star = [delta_star; temp(2,:)];
    end
end
    
    % Display size of final adjustments
    size(gamma_star)
    
    % Apply the empirical Bayes adjustments to the data
    fprintf('[combat] Adjusting the Data\n')
    bayesdata = s_data;
    j = 1;
    
    
    %%% To be corrected later so that it saves time, if EB == False, set gamma
    %%% 
    %%% start and delta star to the gamma and delta hat values for each feature
    
    if UseEB == false
        disp('Discounting the EB adjustments and using Raw estimates, this is not advised')
        delta_star = delta_hat;
        gamma_star = gamma_hat;
    end
    
    % Loop to check the delta and gamma corrections
    if DeltaCorrection==true
        if GammaCorrection == true
            for i = 1:n_batch
                indices = batches{i};
                bayesdata(:, indices) = (bayesdata(:, indices) - (batch_design(indices, :) * gamma_star)') ./(sqrt(delta_star(j, :))' * repmat(1, 1, n_batches(i)));
                j = j + 1;
            end
        elseif GammaCorrection== false
             for i = 1:n_batch
                indices = batches{i};
                bayesdata(:, indices) = (bayesdata(:, indices)) ./(sqrt(delta_star(j, :))' * repmat(1, 1, n_batches(i)));
                j = j + 1;
             end
        end
    elseif DeltaCorrection == false
        if GammaCorrection == true
            for i = 1:n_batch
                indices = batches{i};
                bayesdata(:, indices) = (bayesdata(:, indices) -(batch_design(indices, :) * gamma_star)');
                j = j + 1;
            end
        elseif GammaCorrection== false
                warning('Both Gamma and delta have been set to false, no ComBat adjustments have been applied')
        end
    
    end
       
    % Transform data back to original scale
    if RegressCovariates == true
        % If regress covariates set to true, omitt covariate effects from the
        % standard mean adjustment
        size(Cov_effects)
        bayesdata = (bayesdata .* (sqrt(var_pooled) * repmat(1, 1, n_array))) + (stand_mean-Cov_effects);
    else
    
        % Else, follow normal ComBat logic
        bayesdata = (bayesdata .* (sqrt(var_pooled) * repmat(1, 1, n_array))) + stand_mean;
    end
    
end
