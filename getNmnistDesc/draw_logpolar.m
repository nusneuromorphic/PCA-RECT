function draw_logpolar(img,point,r_min,r_max,nbins_theta,nbins_r)
%SCDRAWPOLAR draw a polar on the center point
%   point           - the center point
%   r_min           - min radius
%   r_max           - max radius
%   nbins_theta     - theta divide
%   nbins_r         - r divide
%   fig_handle      - draw the diagram on which figure
h1=imshow(img,[]);
set(h1,'alphadata',0.6);
gca;
hold on;

% plot(samp(1,:)',samp(2,:)','r.');
plot(point(1),point(2),'ko');

r_bin_edges=logspace(log10(r_min),log10(r_max),nbins_r);

% draw circles
th = 0 : pi / 50 : 2 * pi;
xunit = cos(th);
yunit = sin(th);
for i=1:length(r_bin_edges)
    line(xunit * r_bin_edges(i) + point(1), ...
                    yunit * r_bin_edges(i) + point(2), ...
        'LineStyle', ':', 'Color', 'k', 'LineWidth', 1.2);
end

% draw spokes
th = (1:nbins_theta) * 2*pi / nbins_theta;
cs = [cos(th);zeros(1,size(th,2))];
sn = [sin(th);zeros(1,size(th,2))];
line(r_max*cs + point(1), r_max*sn + point(2),'LineStyle', ':', ...
    'Color', 'k', 'LineWidth', 1);

axis equal;
axis off;
hold off;
