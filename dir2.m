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
    
    k    = 1;

    while k <= length(listing)
        if any(strncmp(listing(k).name, '.', 1)) 
            inds(end + 1) = k;
            
        end
        k = k + 1;
    end

    listing(inds) = [];
end