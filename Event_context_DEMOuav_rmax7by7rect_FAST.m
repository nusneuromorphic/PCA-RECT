clear;
close all;

warning('off', 'MATLAB:MKDIR:DirectoryExists');

tic;
% Add VLFeat toolbox to MATLAB working path
run('vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup.m');
addpath(genpath('getNmnistDesc'));

% Put all the folders (classes) in a single folder
train_dataset_path = '../N-SOD Dataset/Train';

filenamesA = dir2(train_dataset_path);

num_classes=numel(filenamesA);
classnames = cell(1, num_classes);
for i=1:num_classes
    classnames{i}=filenamesA(i).name;
end

% create training labels later
train_label=[];

num_test= 2;
filename = 'traindata_binTD4cl_CYnewobs.dat';
testfiles_save = 'testfiles_binTD4cl_CYnewobs';

try
  canUseGPU = parallel.gpu.GPUDevice.isAvailable;
catch ME
  canUseGPU = false;
end

param.descsize = 5; % denotes a N by N region sampled from the subsampled count matrix
param.queuesize = 5000;
param.countmatsubsamp = 2; % a 2 by 2 cell region for the count matrix
param.minNumEvents = 200; % wait for this many events before getting desc from queue

disp(param);

if exist(filename, 'file') ~= 2
    % Training and testing set filenames
    for i=1:num_classes
        
        folder_path = fullfile(train_dataset_path,classnames{i});
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        
        randomize_files{i} = randperm(numel(subfolder_names));
        num_train = numel(randomize_files{i});
        index= [];
        
        for ii = randomize_files{i}
            filepath = fullfile(folder_path, subfolder_names{ii});
            img = read_linux(filepath, 0);
            if size(img.x,2) > 5000
                index = [index ii];
            end
            if numel(index) > 80
                break
            end
        end
        train_filenames{i} = vl_colsubset(subfolder_names(index),num_train, 'beginning');
        test_filenames{i} =  vl_colsubset(subfolder_names(index),num_test, 'beginning');
    end
    
    fid = fopen(filename, 'w');
    
    % Write into file
    for i = 1: numel(train_filenames)
        classi = train_filenames{i};
        for j = 1: numel(classi)
            classi_objectj = classi{j};
            fprintf(fid, '%s\n', classi_objectj );
        end
    end
    save(testfiles_save,'test_filenames');
end
fid = fopen(filename, 'r');
A = textscan(fid,'%s', 'Delimiter','\n');
all_filenames= A{1};
fclose(fid);

load(testfiles_save);

% TD Noise Filter
% Assumes Garrick's AER functions are available (www.garrickorchard.com)
us_time_filter = 5e3; % in micro-seconds

% Debug purpose --
training_desc_done = 0; % for debug purposes only
svmtraining_done = 0;
testing_desc_done = 0;

% The training phase is here so that it is easy to change parameters
% and check the code of the FPGA implmentation. The testing stage will be
% made modular like how the FPGA implementation is intended to be, fixed
% precision and memory constraints closely followed.

count = 0;
train_desc_savefolder = './ECtraindata_DEMOrectFAST_ver2_5by5_PCA/';
mkdir(train_desc_savefolder);
fprintf('\n\n------------------------------Starting Training Stage------------------------------\n\n');
if training_desc_done == 0
    fprintf('Getting events and saving descriptors for training...\n\n');
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
            filename = all_filenames{count};
            filepath = fullfile(train_dataset_path,classi_name, filename);
            
            TD = read_linux(filepath, 0);
            TD = FilterTD(TD, us_time_filter);
            
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
            fprintf('File num: %d\n\n', count);
        end
    end
    
    save('./NDEMO4desc_5by5subsamp2x2_ustime5e3_PCA.mat',...
        'train_label','trainimage_sizes','-v7.3');
    
    if canUseGPU == 1
        poolobj = gcp('nocreate');
        delete(poolobj);
    end
else
    fprintf('Loading descriptors...\n\n');
    load('./NDEMO4desc_5by5subsamp2x2_ustime5e3_PCA.mat');
end

histopts.num_bins = 150; % codebook size
histopts.numSpatialX = [1] ; % ignore but required for code, if [1 2] performs spatial pyramid matching (not needed)
histopts.numSpatialY = [1] ; % ignore

% SVM params
svmOpts.C = 10 ;
svmOpts.biasMultiplier = 1 ;

pcadims = 25;
fprintf('pcadims = %d\n\n', pcadims);
% Build the codebook
clearvars model net
model_str_stringname = 'modelTD4cl_DEMO_FASTver2nonorm_PCA';
fprintf("Loading previously generated codebook...");
try
    load(['./ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(pcadims) num2str(param.countmatsubsamp) num2str(param.descsize)]);
    model_done = 1;
    fprintf("   Found!\n\n");
catch
    model_done = 0;
    fprintf("   Not Found!\n\n");
end

if model_done == 0
    fprintf("Generating codebook...\n\n");
    [train_data, loctrain_label, new_train_label] = readDescs(train_desc_savefolder,8,'nonorm');
    traindata_befpca = single([train_data.desc]);
    attr_means = mean(traindata_befpca');
    
    [coeff, score, latent] = pca(traindata_befpca');
    traindata_afterpca = score(:,1:pcadims)';
    [model.vocab, model.assoc] = vl_kmeans(vl_colsubset(single(traindata_afterpca), 1e6), histopts.num_bins, 'verbose','algorithm', 'ANN') ;
    model.kdtree = vl_kdtreebuild(model.vocab, 'Distance','L1') ;
    model.vocab = double(model.vocab);
    clearvars traindata_afterpca traindata_befpca
    mkdir('./ECtrainmodels/');
    save(['./ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(pcadims) num2str(param.countmatsubsamp) num2str(param.descsize)], ...
        'model', 'attr_means', 'coeff');
    disp(' ');
end

% For possbile integer comparisons on FPGA
model.kdtree.trees.nodes.splitThreshold = round(model.kdtree.trees.nodes.splitThreshold);

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
                filename = all_filenames{count};
                
                filepath=fullfile(train_desc_savefolder, classname, [filename(1:end-4) '.mat']);
                fprintf('Binning: %s\n', filepath);
                [train_data, loctrain_label, label] = readDescs(filepath, [], 'nonorm');
                
                traindata_befpca = single([train_data.desc]);
                traindata_befpca = bsxfun(@minus, traindata_befpca, attr_means');
                
                traindata_afterpca = coeff(:,1:pcadims)'*traindata_befpca;
                
                start_chunk = 1;
                end_chunk = event_chunk_period;
                for event_c = 1:ceil(size(train_data.frames,2)/event_chunk_period)
                    count_sep= count_sep + 1;
                    try
                        desc = traindata_afterpca(:, start_chunk:end_chunk);
                        frames = train_data.frames(:,start_chunk:end_chunk);
                    catch
                        desc = traindata_afterpca(:, start_chunk:end);
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
        save('svvmodel_DEMO_FASTver2_150codebok_5by5_NONORM_PCA.mat','svmmodel');
        
    else
        fprintf("Loading previously generated SVM model...\n\n");
        load('svvmodel_DEMO_FASTver2_150codebok_5by5_NONORM_PCA.mat');
    end
    
    %% -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % testing stage
    count = 0;
    test_label = [];
    test_desc_savefolder = './ECtestdata_DEMOrect_FASTver2_5by5_PCA/';
    mkdir(test_desc_savefolder);
    if testing_desc_done == 0
        fprintf('\nGetting events and saving descriptors for testing...\n');
        for class_i=1:num_classes
            classi_name = cell2mat(classnames(class_i));
            test_classi = test_filenames{class_i};
            mkdir(fullfile(test_desc_savefolder, classi_name));
            for image_classi = 1:numel(test_classi)
                count = count + 1;
                filename = test_classi{image_classi};
                filepath = fullfile(train_dataset_path,classi_name, filename);
                
                savepath_file = fullfile(test_desc_savefolder, classi_name, filename(1:end-4));
                
                TD = read_linux(filepath, 0);
                TD = FilterTD(TD, us_time_filter);
                
                testimage_sizes(count,:) = [max(TD.y) max(TD.x)];
                
                try
                    load(savepath_file);
                catch
                    [frames, desc] = getDescs_RECT_FPGA(TD, param);
                    disp(savepath_file);
                    save(savepath_file,'frames','desc');
                end
                
                test_label = [test_label class_i];
                fprintf('File num: %d\n\n', count);
            end
        end
        testing_desc_done =1;
        save('./NDEMO4testdesc_5by5subsamp2x2_ustime5e3_PCA.mat',...
            'test_label','testimage_sizes','-v7.3');
        
        if canUseGPU == 1
            poolobj = gcp('nocreate');
            delete(poolobj);
        end
    else
        disp('Loading test descriptors...'); % needs modification of code if desc_done=0
        load('./NDEMO4testdesc_5by5subsamp2x2_ustime5e3_PCA.mat');
    end
    
    %% ---------------------------------------------------------------------------
    %---------------------------------------------------------------------------
    % Testing phase
    
    count = 0;
    count_sep =0;
    save_for_majvot = [];
    test_label = [];
    fprintf('\n\n------------------------------Starting Testing Stage------------------------------\n\n');
    for class_i=1:num_classes
        classi_name = cell2mat(classnames(class_i));
        test_classi = test_filenames{class_i};
        for image_classi = 1:numel(test_classi)
            count = count + 1;
            filename = test_classi{image_classi};
            filepath = fullfile(test_desc_savefolder, classi_name, [filename(1:end-4) '.mat']);
            
            [test_data, loctest_label, testlabel] = readDescs(filepath, [], 'nonorm');
            
            testdata_befpca = single([test_data.desc]);
            testdata_befpca = bsxfun(@minus, testdata_befpca, attr_means');
            
            testdata_afterpca = coeff(:,1:pcadims)'*testdata_befpca;
            
            start_chunk = 1;
            end_chunk = event_chunk_period;
            for event_c = 1:ceil(size(test_data.frames,2)/event_chunk_period)
                count_sep= count_sep + 1;
                try
                    desc = testdata_afterpca(:, start_chunk:end_chunk);
                    frames = test_data.frames(:,start_chunk:end_chunk);
                catch
                    desc = testdata_afterpca(:, start_chunk:end);
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
                start_chunk = end_chunk + 1;
                end_chunk = end_chunk + event_chunk_period;
                
                if end_chunk > size(test_data.frames,2)
                    end_chunk = size(test_data.frames,2);
                end
                
            end
            save_for_majvot(count) = count_sep;
            fprintf('File num: %d\n\n', count);
        end
    end
    
    error_locs=find(obj_cat(:,2)'~=test_label);
    error_per=length(error_locs);
    accuracy(repeat)= 100*(numel(test_label)-  error_per)/numel(test_label)
end
toc
