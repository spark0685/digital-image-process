%% ����������������˲��㷨���� myAverage.m��myMedian.m��Ȼ�󽫱��������еı��� 
%% filename����Ϊ'circuit'������main_filter.m�����ɵõ����
%% EEIS Department, USTC
clc; 
clear;
close all;

%% ��ȡͼ��
filename = 'circuit'; % �ܵ�����������Ⱦ�ĵ�·��X����ͼ��
im = imread([filename, '.jpg']);

%% ��ͼ����о�ֵ�˲�
im_a = myAverage(im);

%% ��ͼ�������ֵ�˲�
im_m = myMedian(im);

%% ��������浽��ǰĿ¼�µ�result�ļ�����
imwrite(im_a, sprintf('result/_%s_a.jpg', filename));
imwrite(im_m, sprintf('result/_%s_m.jpg', filename));

%% ��ʾ���
figure(1); 
subplot(131); imshow(im); title('ԭͼ'); axis on
subplot(132); imshow(im_a); title('��ֵ�˲�'); axis on
subplot(133); imshow(im_m); title('��ֵ�˲�'); axis on
