load parallel_projections_filtered.mat; %defines proj_data

N = 512;
proj_len = size(proj_data, 1);
central_det = 367;
image = zeros(N, N);
[rx ry] = meshgrid(-(N-1)/2:(N-1)/2);
dtheta = pi/size(proj_data, 2);
Tdet = 1.25;
Tobj = 1.125;
dets = ((0:(proj_len-1))- central_det)*Tdet;
for p = 1:size(proj_data, 2)
    theta = (p-1) * dtheta;
    tloc = (-rx * sin(theta) + ry * cos(theta))*Tobj;
    img_up = interp1(dets, proj_data(:, p), tloc, 'linear');
    image = image + reshape(img_up, [N N]);
end;
imagesc(image);
colormap(gray);