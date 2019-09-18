from numpy import *
def sigmoid(inx):
	return 1.0/(1 + exp(-inx))

def loaddata(filename): #加载文件函数
	fp = open(filename)
	data = []
	labels = []
	for i in fp.readlines():#readlines读取文件，i指单行
		a = i.strip('\n').replace(',',' ').split() #去多余字符，分割
		a=list(map(eval,a))
		data.append([float(j) for j in a[:21]])
		labels.append(float(a[-1]))
	return data,labels

def logistic(dataset,labels,num=500):
	m,n = shape(dataset)
	datamatrix = mat(dataset)#属性矩阵
	labelmatrix = mat(labels).transpose()#标签列向量
	weights = ones((n,1))
	print(ones(n))
	alpha = 0.001
	for i in range(num):
		h = sigmoid(datamatrix * weights)
		error = labelmatrix - h
		weights = weights + alpha * datamatrix.transpose() * error#datamatrix.transpose()相当于Xj(某j列变量)
	return weights

def Stochastic_gradient_rise(datamatrix,labels,num = 150): #随机梯度上升法进行模型优化
	m,n = shape(datamatrix)
	weights = ones(n)
	for j in range(num):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(datamatrix[randIndex]*weights))
			error = labels[randIndex] - h
			weights = weights + alpha * error * datamatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

def test1(dataset,labels,weights):
	rightcount = 0
	for i in range(len(dataset)):
		prob = sigmoid(sum(dataset[i] * weights))
		if prob > 0.5:
			a = 1
		else:
			a = 0
		print("the test1 result is:%d %d"%(a,labels[i]))
		if a is int(labels[i]):
			rightcount += 1
	rightrate = rightcount/len(dataset)
	errorate = 1 - rightrate
	return errorate,rightrate

def test2(dataset,labels,weights):
	dataMatrix = mat(dataset)#mat函数可以将目标数据的类型转换为矩阵（matrix）
	prob = sigmoid(dataMatrix * weights)
	result = []
	for i in prob:
		if i>0.5:
			result.append(1)
		else:
			result.append(0)
	rightcount = 0
	for i in range(len(result)):
		if result[i] is int(labels[i]):
			rightcount += 1
			print("the test2 result is:%d %d"%(result[i],labels[i]))
		else:
			print("the test2 result is:%d %d"%(result[i],labels[i]))
	rightrate = rightcount/len(labels)
	errorate = 1 - rightrate
	return errorate,rightrate


#训练过程
dataset,labels = loaddata('data_banknote_authentication_TrainData.txt')
print("shape of dataset:") #数据集维度
print(shape(dataset))
print("shape of array(dataset)") #在numpy中维度
print(shape(array(dataset)))

#通过训练更新权值
weights1 = Stochastic_gradient_rise(array(dataset),labels,500)
weights2 = logistic(dataset,labels,500)
#print weights
print("weights2:")
print(weights2)

#利用测试集测试
testdata,testlabels = loaddata('data_banknote_authentication_TestingData.txt')
errorate1,rightrate1 = test1(testdata,testlabels,weights1)
errorate2,rightrate2 = test2(testdata,testlabels,weights2)
print('logistic回归错误率为：%f\n正确率为：%f'%(errorate1,rightrate1))
print('随机梯度上升优化后错误率为：%f\n正确率为：%f'%(errorate2,rightrate2))
