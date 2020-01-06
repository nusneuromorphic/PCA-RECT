%Return: a collection of array of spike coordinates to generate histogram
function arr = getSpikeArr(TD)
    %fileName = '00002.bin';
    
    arr = {};
    i = 200; minId = 0;
    count = 1;
    while TD.ts(i)<=105000
        maxId = i;
        coordArr = getCoordArr(maxId, minId, TD.x, TD.y);
        arr{1, count} = coordArr; 
        i = i+1;
        count = count + 1;
    end

    minId = i-1;
    i = i+199;
    count = 1;
    while TD.ts(i)<=210000
        maxId = i;
        coordArr = getCoordArr(maxId, minId, TD.x, TD.y);
        arr{2, count} = coordArr;
        i = i+1;
        count = count + 1;
    end

    minId = i-1;
    i = i+199;
    count = 1;
    while i<= length(TD.ts)
        maxId = i;
        coordArr = getCoordArr(maxId, minId, TD.x, TD.y);
        arr{3,count} = coordArr;
        i = i+1;
        count = count + 1;
    end
%     for j=1:(i-1)
%         disp(arr{j}); pause;
%     end
end

function coordArr = getCoordArr(maxId, minId, x, y)
    coordArr = [];
    while maxId>minId
        coordArr = [coordArr [x(maxId) y(maxId)]];
        maxId = maxId - 1;
    end
    coordArr = reshape(coordArr, 2, [])';
end

function writeFile(fileName, var)
    writetable(struct2table(var), fileName);
end

function TD = readSample(fileName)
    TD = Read_Ndataset(fileName);
    TD = stabilize(TD);
end
