function [train_data, train_label, rndIdxArr] = readDescs(path, numTD, norm)
    rndIdxArr = [];

    if (nargin<3)
        norm = 'L1';
    end
    
    [~, ~, ext] = fileparts(path);
    if (isequal('.mat', ext))
        file = load(path);
        if isequal(norm,'L1')
            file.desc = l1norm(file.desc);
        elseif isequal(norm,'L2')
            file.desc = l2norm(file.desc);
        elseif isequal(norm,'nonorm')
            file.desc = file.desc; 
        end
        train_data.frames = file.frames;
        train_data.desc = file.desc;
        %loctrain_label = [file.loctrain_label];

            parts = strsplit(path, '/');
        if isequal(cell2mat(parts),path)
            parts = strsplit(path, '\');
        end
        train_label = parts{end-1};
        return
    end
    
    count = 0; train_label = [];
    %loctrain_label = [];
	classnames = dir2(path);
    num_classes = length(classnames);
    nSample = ceil(numTD/num_classes);
    for i=1:num_classes
        classname = classnames(i).name;
        fprintf("Getting descriptors from: %s\n", fullfile(path, classname));
        datainclass = dir2(fullfile(path, classname));
        rndIdx = randperm(length(datainclass));
        rndIdxArr(i).rndIdx = rndIdx;
        samples = datainclass(rndIdx(1:nSample), :);
        for j=1:length(samples)
            count = count + 1;
            filename = fullfile(path, classname, samples(j).name);
            file = load(filename);

            if isequal(norm,'L1')
                file.desc = l1norm(file.desc);
            elseif isequal(norm,'L2')
                file.desc = l2norm(file.desc);
            elseif isequal(norm,'nonorm')
                file.desc = file.desc; 
            end
            train_data(count).frames = file.frames;
            train_data(count).desc = file.desc;
%             train_data(count).image_sizes = file.image_sizes;
            %loctrain_label = [loctrain_label file.loctrain_label];
            train_label= [train_label ones(1,size(train_data(count).frames,2)) * i];
        end  
    end
    disp(' ');
end

function output = l2norm(input)
    n = sqrt(sum(input.^2,1));
    output = bsxfun(@rdivide, input, n);
end

function output = l1norm(input)
    input = single(input); 
    n = sum(abs(input), 1);
    output = bsxfun(@rdivide, input, n);
end
