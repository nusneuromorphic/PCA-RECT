clear;
close all;

% Add VLFeat toolbox to MATLAB working path
run('vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup.m');
addpath(genpath('getNmnistDesc'));

% Put all the folders (classes) in a single folder
train_dataset_path = '../N-SOD Dataset/Train';
test_dataset_path = '../N-SOD Dataset/Test';

filenamesA = dir2(train_dataset_path);

num_classes=numel(filenamesA);
classnames = cell(1, num_classes);
for i=1:num_classes
    classnames{i}=filenamesA(i).name;
end
% create training labels later
train_label=[];

param.descsize = 7; % denotes a N by N region sampled from the subsampled count matrix
param.queuesize = 5000;
param.countmatsubsamp = 2; % a 2 by 2 cell region for the count matrix
param.minNumEvents = 200; % wait for this many events before getting desc from queue

disp(param);

% TD Noise Filter
% Assumes Garrick's AER functions are available (www.garrickorchard.com)
us_time_filter = 5e3; % in micro-seconds
refractory_period = 1e3;

% Debug purpose --
training_desc_done = 0; % for debug purposes only
svmtraining_done = 0;
testing_desc_done = 0;

try
  canUseGPU = parallel.gpu.GPUDevice.isAvailable;
catch ME
  canUseGPU = false;
end

% The training phase is here so that it is easy to change parameters
% and check the code of the FPGA implementation. The testing stage will be
% made modular like how the FPGA implementation is intended to be, fixed
% precision and memory constraints closely followed.

count = 0;
train_desc_savefolder = './Recognition_trainfiles/D5DEMOrectFASTnoPCA_splitaug/';
mkdir(train_desc_savefolder);
if training_desc_done == 0
    for class_i=1:num_classes
        classi_name = cell2mat(classnames(class_i));
        folder_path = fullfile(train_dataset_path,classi_name);
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        mkdir(fullfile(train_desc_savefolder, classi_name));
        for train_classi = 1:numel(subfolder_names)
            count = count  + 1;
            filename = subfolder_names{train_classi};
            filepath = fullfile(train_dataset_path,classi_name, filename);
            
            TD = read_linux(filepath, 0);
            TD = ImplementRefraction(TD, refractory_period);
            TD = FilterTD(TD, us_time_filter);
            
            if numel(TD.x) < 3000
                TD = read_linux(filepath, 0);
                TD = ImplementRefraction(TD, refractory_period);
            end
            
            trainimage_sizes(count,:) = [max(TD.y) max(TD.x)];
            savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
            
            try
                load(savepath_file);
            catch
                [frames, desc] = getDescs_RECT_FPGA(TD, param);
                
                savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
                disp(savepath_file);
                save(savepath_file,'frames','desc');
            end
            
            train_label = [train_label class_i];
            disp(count);
            
        end
    end
    
    save('./Recognition_trainfiles/D5DEMO4splitAUGdesc_7x7subsamp2x2_ustime5e3.mat',...
        'train_label','trainimage_sizes','-v7.3');
    
    if canUseGPU == 1
        poolobj = gcp('nocreate');
        delete(poolobj);
    end
else
    disp('Loading descrs...');
    load('./Recognition_trainfiles/D5DEMO4splitAUGdesc_7x7subsamp2x2_ustime5e3.mat');
end

histopts.num_bins = 150; % codebook size
histopts.numSpatialX = [1] ; % ignore but required for code, if [1 2] performs spatial pyramid matching (not needed)
histopts.numSpatialY = [1] ; % ignore

% SVM params
svmOpts.C = 10 ;
svmOpts.biasMultiplier = 1 ;

% Build the codebook
clearvars model net
model_str_stringname = 'modelTD4cl_D5DEMOsplitAUG_FASTnoPCA';
try
    load(['./Recognition_trainfiles/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)]);
    model_done = 1;
catch
    model_done = 0;
end

if model_done == 0
    [train_data, loctrain_label, new_train_label] = readDescs(train_desc_savefolder,20, 'nonorm');
    [model.vocab, model.assoc] = vl_kmeans(vl_colsubset(single([train_data.desc]), 4e6), histopts.num_bins, 'verbose','algorithm', 'ANN') ;
    model.kdtree = vl_kdtreebuild(model.vocab, 'Distance','L1') ;
    model.vocab = double(model.vocab);
    mkdir('./Recognition_trainfiles/ECtrainmodels/');
    save(['./Recognition_trainfiles/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)],'model');
end

% For possbile integer comparisons on FPGA
model.kdtree.trees.nodes.splitThreshold = floor(model.kdtree.trees.nodes.splitThreshold);

% Get hists
for repeat= 1
    clearvars hists*
    
    train_label = [];
    count_sep = 0;
    count = 0;
    event_chunk_period = 1e4; % get a histogram for classification every 10k events, same for testing
    if svmtraining_done == 0
        for class_i=1:num_classes
            classname=classnames{class_i};
            folder_path = fullfile(train_dataset_path,classname);
            subfolder_names = dir(folder_path);
            isdir_tags = [subfolder_names.isdir];
            subfolder_names = subfolder_names(~isdir_tags);
            subfolder_names = {subfolder_names.name};
            for j=1:numel(subfolder_names)
                count = count + 1;
                filename = subfolder_names{j};
                
                filepath=fullfile(train_desc_savefolder, classname, [filename(1:end-4) '.mat']);
                disp(filepath);
                [train_data, loctrain_label, label] = readDescs(filepath, [], 'nonorm');
                
                start_chunk = 1;
                end_chunk = event_chunk_period;
                for event_c = 1:ceil(size(train_data.frames,2)/event_chunk_period)
                    count_sep= count_sep + 1;
                    try
                        desc = train_data.desc(:, start_chunk:end_chunk);
                        frames = train_data.frames(:,start_chunk:end_chunk);
                    catch
                        desc = train_data.desc(:, start_chunk:end);
                        frames = train_data.frames(:,start_chunk:end);
                    end
                    hists{count_sep} = get_imagehist(model,trainimage_sizes(count,:),...
                        double(frames), double(desc), histopts, 'nonorm');
                    train_label = [train_label class_i];
                    start_chunk = end_chunk + 1;
                    end_chunk = end_chunk + event_chunk_period;
                    
                    if end_chunk > size(train_data.frames,2)
                        end_chunk = size(train_data.frames,2);
                    end
                end
            end
        end
        
        hists = cat(2, hists{:}) ;
        hists=cat(1,hists);
        psix = hists; % kernel removed, above line applies kernel
        
        %SVM Calculation
        lambda = 1 / (svmOpts.C * length(train_label)) ;
        w = [] ;
        for ci = 1:length(unique(train_label))
            perm = randperm(length(train_label)) ;
            y = 2 * (train_label == ci) - 1 ;
            [w(:,ci) b(:, ci) info(ci)] = vl_svmtrain(psix(:,perm), ...
                y(perm), lambda, ...
                'MaxNumIterations', 50/lambda, ...
                'BiasMultiplier', svmOpts.biasMultiplier) ;
        end
        
        svmmodel.b = b ;
        svmmodel.w = w;
        save('./Recognition_trainfiles/svvmodel_D5DEMOsplitAUG_FPGAver2_150codebokNONORM_7by7_noPCA.mat','svmmodel');
    else
        load('./Recognition_trainfiles/svvmodel_D5DEMOsplitAUG_FPGAver2_150codebokNONORM_7by7_noPCA.mat');
    end
    
    %% -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % testing stage
    
    count = 0;
    test_label = [];
    test_desc_savefolder = './Recognition_trainfiles/D5DEMOrect_FPGAver2_testsplitaug/';
    mkdir(test_desc_savefolder);
    if testing_desc_done == 0
        for class_i=1:num_classes
            classi_name = cell2mat(classnames(class_i));
            folder_path = fullfile(test_dataset_path,classi_name);
            subfolder_names = dir(folder_path);
            isdir_tags = [subfolder_names.isdir];
            subfolder_names = subfolder_names(~isdir_tags);
            subfolder_names = {subfolder_names.name};
            mkdir(fullfile(test_desc_savefolder, classi_name));
            for image_classi = 1:numel(subfolder_names)
                count = count + 1;
                filename = subfolder_names{image_classi};
                filepath = fullfile(test_dataset_path,classi_name, filename);
                
                savepath_file = fullfile(test_desc_savefolder, classi_name, filename(1:end-4));
                
                TD = read_linux(filepath, 0);
                TD = ImplementRefraction(TD, refractory_period);
                TD = FilterTD(TD, us_time_filter);
                
                if numel(TD.x) < 3000
                    TD = read_linux(filepath, 0);
                    TD = ImplementRefraction(TD, refractory_period);
                end
                
                testimage_sizes(count,:) = [max(TD.y) max(TD.x)];
                
                try
                    load(savepath_file);
                catch
                    
                    [frames, desc] = getDescs_RECT_FPGA(TD, param);
                    disp(savepath_file);
                    save(savepath_file,'frames','desc');
                end
                
                test_label = [test_label class_i];
                disp(count);
            end
        end
        testing_desc_done =1;
        save('./Recognition_trainfiles/D5DEMO4splitAUGtestdesc_7by7subsamp2x2_ustime5e3_noPCA.mat',...
            'test_label','testimage_sizes','-v7.3');
        
        if canUseGPU == 1
            poolobj = gcp('nocreate');
            delete(poolobj);
        end
    else
        disp('Loading test descrs...'); % needs modification of code if desc_done=0
        load('./Recognition_trainfiles/D5DEMO4splitAUGtestdesc_7by7subsamp2x2_ustime5e3_noPCA.mat');
    end
    
    %% ---------------------------------------------------------------------------
    % Testing phase
    
    count = 0;
    count_sep =0;
    save_for_majvot = [];
    test_label = [];
    for class_i=1:num_classes
        classi_name = cell2mat(classnames(class_i));
        folder_path = fullfile(test_dataset_path,classi_name);
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        for image_classi = 1:numel(subfolder_names)
            count = count + 1;
            filename = subfolder_names{image_classi};
            filepath = fullfile(test_desc_savefolder, classi_name, [filename(1:end-4) '.mat']);
            
            [test_data, loctest_label, testlabel] = readDescs(filepath, [], 'nonorm');
            
            % Read original TD for display
            filepath = fullfile(test_dataset_path,classi_name, filename);
            TD = read_linux(filepath, 0); % the bin files are stored with refraction and FilterTD already
            
            TD.x = TD.x(:);
            TD.y = TD.y(:);
            TD.ts = TD.ts(:);
            TD.p = TD.p(:);
            
            % Saved bin files are not with refraction and FilterTD
            TD = ImplementRefraction(TD, refractory_period);
            TD = FilterTD(TD, us_time_filter);
            
            if numel(TD.x) < 3000
                TD = read_linux(filepath, 0);
                TD = ImplementRefraction(TD, refractory_period);
            end
            
            temp_count_matrix = zeros(testimage_sizes(1,1),testimage_sizes(1,2));  % image_sizes is in [y x] order already
            
            start_chunk = 1;
            end_chunk = event_chunk_period;
            for event_c = 1:ceil(size(test_data.frames,2)/event_chunk_period)
                count_sep= count_sep + 1;
                try
                    desc = test_data.desc(:, start_chunk:end_chunk);
                    frames = test_data.frames(:,start_chunk:end_chunk);
                catch
                    desc = test_data.desc(:, start_chunk:end);
                    frames = test_data.frames(:,start_chunk:end);
                end
                
                
                hists = get_imagehist(model,testimage_sizes(count,:),...
                    double(frames), double(desc), histopts, 'nonorm');
                
                psix = hists;
                
                %SVM Classification
                scores = svmmodel.w' * psix + svmmodel.b' * ones(1,size(psix,2)) ;
                [obj_cat(count_sep,1), obj_cat(count_sep,2)] = max(scores);
                %                     keyboard
                test_label = [test_label class_i];
                
                % Display purposes only
                if class_i ~= obj_cat(count_sep,2)
                    try
                        all_locations_x = TD.x(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: event_chunk_period + (event_c - 1) * event_chunk_period + param.minNumEvents);
                        all_locations_y = TD.y(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: event_chunk_period + (event_c - 1) * event_chunk_period + param.minNumEvents);
                    catch
                        all_locations_x = TD.x(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: end);
                        all_locations_y = TD.y(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: end);
                    end
                    all_linear_indices = sub2ind(size(temp_count_matrix), all_locations_y, all_locations_x);
                    temp_count_matrix(all_linear_indices) = temp_count_matrix(all_linear_indices) + 1;
                    figure(1),
                    imshow(temp_count_matrix)
                    title(sprintf('Classified as %s',cell2mat(classnames(obj_cat(count_sep,2)))));
                    drawnow()
                    pause(2)
                    temp_count_matrix = zeros(size(temp_count_matrix));
                end
                start_chunk = end_chunk + 1;
                end_chunk = end_chunk + event_chunk_period;
                
                if end_chunk > size(test_data.frames,2)
                    end_chunk = size(test_data.frames,2);
                end
                
            end
            save_for_majvot(count) = count_sep;
            disp(count);
        end
    end
    
    error_locs=find(obj_cat(:,2)'~=test_label)
    error_per=length(error_locs)
    accuracy(repeat)= 100*(numel(test_label)-  error_per)/numel(test_label)
    confusionmat(single(test_label), single(obj_cat(:,2)));
end