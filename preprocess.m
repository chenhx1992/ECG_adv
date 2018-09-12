clc;
clear all;
close all;
add = '.\training_raw\';
n = 8528;
id = [1:1:n];
len = zeros(1,length(n));
% min_len = 400;
% max_len = 0;
for i = 1:n

name = pad(num2str(id(i)),5,'Left','0');
raw_ecg = importdata([add,'A',name,'.mat']);
% figure;
% plot(raw_ecg)
Fs = 300;
% spectrogram(raw_ecg,tukeywin(64,0.25),32,64,Fs,'yaxis');
[s,f,t,p] = spectrogram(raw_ecg,tukeywin(64,0.25),32,64,Fs,'yaxis');

norm_p = abs(p);
norm_p = pow2db(norm_p);
norm_p = (norm_p - min(norm_p(:)))./(max(norm_p(:)) - min(norm_p(:)));
len(i) = size(norm_p,2);
% if (len(i))
%     t(end)
% end;
% filename = [add,'A',name,'.csv']
% csvwrite(filename,norm_p);
end;
% figure;
% h = surf(norm_p);
% view(0, 90);
% axis tight
% set(h,'linestyle','none');
% set(gca,'YScale','log')
% grid off;
% colorbar;
% 





