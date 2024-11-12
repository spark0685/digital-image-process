%% �������ֱ��ͼ���⻯���� myHisteq.m��Ȼ�󽫱��������еı��� 
%% filename���á�bridge��������main_histeq.m�����ɵõ�ֱ��ͼ���⻯��Ľ��
%% EEIS Department, USTC
clc; 
clear;
close all;

%% ��ȡͼƬ
filename = 'bridge'; %����ͼ��1
im = imread([filename, '.jpg']);
n_1 = 2;
n_2 = 64;
n_3 = 256;

%% ��ͼ�����ֱ��ͼ���⻯
im_histeq_1 = myHisteq(im, n_1);
im_histeq_2 = myHisteq(im, n_2);
im_histeq_3 = myHisteq(im, n_3);

%% ��������浽��ǰĿ¼�µ�result�ļ�����
imwrite(im_histeq_1, sprintf('result/_%s_eq_%.d.jpg', filename, n_1));
imwrite(im_histeq_2, sprintf('result/_%s_eq_%.d.jpg', filename, n_2));
imwrite(im_histeq_3, sprintf('result/_%s_eq_%.d.jpg', filename, n_3));

%% ��ʾ���
figure(1); 
subplot(221); imshow(im); title('ԭͼ'); axis on
subplot(222); imshow(im_histeq_1); title('ֱ��ͼ���⻯, n=2'); axis on
subplot(223); imshow(im_histeq_2); title('ֱ��ͼ���⻯, n=64'); axis on
subplot(224); imshow(im_histeq_3); title('ֱ��ͼ���⻯, n=256'); axis on
