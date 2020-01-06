%Param: directory that contains all binary files for each label
%Return: loop through all the files in the input directory
function data = getDescPerLabel(path)
    %path = '/Users/NgocAnh/Documents/yr4sem2/CG4001/FYP_new/input';
    output = {};
    fileList = dir2(path);
    for i=1:length(fileList)
%         disp(fileList(i).name); pause;
        spikeArr = getSpikeArr(fullfile(path,fileList(i).name));
        [data(i).frames, data(i).desc] = getLogHistArr(spikeArr);
    end
end

function listing = dir2(varargin)
    if nargin == 0
        name = '.';
    elseif nargin == 1
        name = varargin{1};
    else
        error('Too many input arguments.')
    end

    listing = dir(name);

    inds = [];
    n    = 0;
    k    = 1;

    while n < 2 && k <= length(listing)
        if any(strcmp(listing(k).name, {'.', '..'}))
            inds(end + 1) = k;
            n = n + 1;
        end
        k = k + 1;
    end

    listing(inds) = [];
end