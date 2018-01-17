'''
Created on Jan 15, 2018
kNN:k nearest neighbors

Input:
            inx: vector to compare to existing data set (1*N)
            data_set: size m data set of known vectors (M*N)
            labels: data set labels (1*M)
            k: number of neighbors to use for comparison (should be\
                an odd number)

Output:
            the most popular class label

@author:
            LiZhi
            lizhimushui@163.com

'''

from os import listdir
from numpy import *
import operator


# 原始数据集
def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类函数定义
def classify0(inx, data_set, labels, k):
    # 计算距离
    data_set_size = data_set.shape[0]  # 训练集行数
    diff_mat = tile(inx, (data_set_size, 1)) - data_set  # 重复\
    # data_set行数的输入数据inX,再减去原始的data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)  # 将差异平方矩阵的行求和
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()  # 返回距离升序\
    # 排序的索引

    # 计算类别
    class_count = {}  # 大括号表示字典
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]  # 按距离从小到大\
        # 返回前k个的labels
        # 利用get(l,0)使得字典按照标签计数，先碰到没有的返回0，\
        # 有的直接返回值，
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 按照class_count里的第2个元素降序,输出的是一个列表，第1维是labels，\
    # 第2维是个数
    sorted_class_count = sorted(class_count.items(), key=operator. \
                                itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 返回排在列表中最高的labels


'''

# 分类函数的运行测试记录
group, labels = create_data_set()
print(classify0([0, 0], group, labels, 4))

'''


# 将文本转换为可处理数据的函数
def file2matrix(filename):
    fr = open(filename)
    array_lines = fr.readlines()
    number_of_lines = len(array_lines)
    return_mat = zeros([number_of_lines, 3])  # 改为了只有一个()
    class_label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()  # 截取掉所有回车字符
        list_from_line = line.split('\t')  # 使用tab字符将上面的得到的\
        # 整行数据分割成1个元素列表
        return_mat[index, :] = list_from_line[0:3]  # 0:3表示0，1，2

        # # 如果需要将文字转化为数字
        # if (list_from_line[-1] == 'largeDoses'):
        #     list_from_line[-1] = 3
        # elif (list_from_line[-1] == 'smallDoses'):
        #     list_from_line[-1] = 2
        # else:
        #     list_from_line[-1] = 1

        class_label_vector.append(list_from_line[-1])  # -1表示\
        # 列表中的最后1列元素，利用负索引将最后1列存储到class_label_vector
        index += 1
    return return_mat, class_label_vector


'''

# 将文本转换为可处理数据的函数的运行测试记录
dating_data_mat,dating_labels = file2matrix('datingTestSet2.txt')
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()  
ax = fig.add_subplot(111)  
ax.scatter(dating_data_mat[:,0],dating_data_mat[:,1],\
15.0*array(int(dating_labels)),15.0*array(int(dating_labels)))
plt.show()  
plt.xlabel('每年获取的飞行常客里程数')
plt.ylabel('玩视频游戏所耗时间百分比') 
ax.set_title('约会数据散点图')  

'''


# 归一化特征值
def autoNorm(data_set):
    min_vals = data_set.min(0)  # 0表示求列的最小值
    max_vals = data_set.max(0)  # 0表示求列的最大值
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]  # 读取矩阵的行数，第一维的维数
    norm_data_set = data_set - tile(min_vals, (m, 1))  # 此处tile表示行数\
    # 重复m次列数重复1次
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # 得到标准化的数值结果
    return norm_data_set, ranges, min_vals


'''

# 运行测试记录
norm_mat,ranges,min_vals = autoNorm(dating_data_mat)

'''


# 分类器针对约会网站的测试代码
def dating_class_test():
    # 测试集所占比例
    our_ratio = 0.10
    # 从文本中读取数据
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    # 极差标准化
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    m = norm_mat.shape[0]  # 总样本数m
    num_test_vecs = int(m * our_ratio)  # 测试样本数
    error_count = 0.0  # 误差初始化
    # 利用k近邻分类函数得到分类结果并输出，计算错误率
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], \
                                      dating_labels[num_test_vecs:m], 3)
        print("The classifier came back with: %s,and the real answer is:%s" \
              % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]): error_count += 1.0
    print("The error number is : %f.\nThe test number is : %f.\n\
The total error rate is : %f" % (error_count, num_test_vecs, error_count / float(num_test_vecs)))


'''

# 分类器针对约会网站的测试记录
dating_class_test()

'''


# 约会网站的预测函数
def classify_person():
    # # 结果输出列表
    # result_list = ['didntLike', 'in small doses', 'in large doses']
    # 键入用户数据
    percent_tats = float(input("Input percentage of time spent playing video games:"))
    ffmiles = float(input("Input frequent flier miles earned per year:"))
    ice_cream = float(input("Input liters of ice cream consumed per year:"))
    # 从文本中读取数据
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    # 极差标准化
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    # 拼成数组用作分类器的输入
    in_arr = array([ffmiles, percent_tats, ice_cream])
    # 将标准化后的数据输入分类器做判断
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, \
                                  dating_labels, 3)
    print("You will probably like this person: %s " % classifier_result )


'''

# 约会网站实际预测测试记录
classify_person()

'''


# 将32*32的图像转换成1*1024的向量
def img2vector(file_name):
    return_vect = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


'''

# 图像转换成向量测试记录
test_vector = img2vector('digits/testDigits/0_13.txt')

'''


# 手写数字识别系统的测试
def hand_writing_class_test():
    our_labels = []
    # 获取训练集目录内容以及文件数m
    training_file_list = listdir('digits/trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        # 下面对原书中的代码进行了修改,发现没有必要引入一个file_str
        # split('*',num)通过指定分隔符*对字符串进行切片，如果参数num 有指定值，\
        # 则仅分隔 num +1 个子字符串，此处分割得到实际标签
        class_num_str = int(file_name_str.split('_')[0])
        # 将实际标签加入到our_labels
        our_labels.append(class_num_str)
        # 将第i个文件从图像转换为向量
        training_mat[i, :] = img2vector('digits/trainingDigits/%s' % file_name_str)
    # 获取测试集目录内容以及文件数m_test
    test_file_list = listdir('digits/testDigits')
    m_test = len(test_file_list)
    # 错误数记录
    error_count = 0.0
    for i in range(m_test):
        file_name_str = test_file_list[i]
        # 下面对原书中的代码进行了修改,发现没有必要引入一个file_str
        # split('*',num)通过指定分隔符*对字符串进行切片，如果参数num 有指定值，\
        # 则仅分隔 num +1 个子字符串，此处分割得到实际标签
        class_num_str = int(file_name_str.split('_')[0])
        # 将测试文件从图像转换为向量
        vector_under_test = img2vector('digits/testDigits/%s' % file_name_str)
        # 调用classify0函数得到k近邻的分类结果
        classifier_result = classify0(vector_under_test, training_mat, our_labels, 3)
        # k近邻算法输出标签和实际标签比较
        print("The classifier came back with: %d,and the real answer is:%d" \
              % (classifier_result, class_num_str))
        if (classifier_result != class_num_str): error_count += 1.0
    print("The error number is : %f.\nThe test number is : %f.\n\
The total error rate is : %f" % (error_count, m_test, error_count / float(m_test)))


'''

# 手写数字识别系统的测试记录
hand_writing_class_test()

'''
