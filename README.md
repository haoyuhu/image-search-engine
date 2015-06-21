---
layout: post
title: OpenCV实现图像搜索引擎(Image Search Engine)
comments: true
---


简单介绍一下[OpenCV](http://opencv.org)。

OpenCV was designed for computational efficiency and with a strong focus on real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing. Enabled with OpenCL, it can take advantage of the hardware acceleration of the underlying heterogeneous compute platform. Adopted all around the world, OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 9 million. Usage ranges from interactive art, to mines inspection, stitching maps on the web or through advanced robotics.

OpenCV（Open Source Computer Vision Library）的计算效率很高且能够完成实时任务。OpenCV库由优化的C/C++代码编写而成，能够充分发挥多核处理和硬件加速的优势。OpenCV有大量技术社区和超过900万的下载量，它的使用范围极为广泛，如人机互动、资源检查、拼接地图等。

## 0.Python+OpenCV实现图像搜索引擎
之前看到谷歌和百度出了图像搜索引擎，查阅了相关资料深入了解了图像搜索引擎的算法原理。一部分参考了[用Python和OpenCV创建一个图片搜索引擎的完整指南](http://python.jobbole.com/80860/)。决定自己实现一个简单的图像搜索引擎，也可以让自己更快地查找mac中的图片。为什么使用**OpenCV+Python**实现图像搜索引擎呢？

* 首先，OpenCV是一个开源的**计算机视觉处理库**，在**计算机视觉**、**图像处理**和**模式识别**中有广泛的应用。接口安全易用，而且**跨平台**做的相当不错，是一个不可多得的计算机图像及视觉处理库。

* 其次，Python的语法更加易用，**贴近自然语言，极为灵活**。虽然计算效率并不高，但快速开发上它远胜于C++或其他语言，引入**pysco能够优化python代码中的循环**，一定程度上缩小与C/C++在计算上的差距。而且图像处理中需要大量的矩阵计算，**引入numpy做矩阵运算**能够降低编程的冗杂度，更多地把精力放在匹配的逻辑上，而非计算的细枝末节。

## 1. 图像搜索原理
图像搜索算法基本可以分为如下步骤：

* **提取图像特征。**如采用**SIFT、指纹算法函数、哈希函数、bundling features算法**等。当然如知乎中所言，也可以**针对特定的图像集群采用特定的模式设计算法**，从而提高匹配的精度。如已知所有图像的中间部分在颜色空间或构图上有显著的区别，就可以加强对中间部分的分析，从而更加高效地提取图像特征。

* **图像特征的存储。**一般**将图像特征量化为数据**存放于索引表中，并存储在外部存储介质中，搜索图片时仅搜索索引表中的图像特征，按匹配程度从高到低查找类似图像。对于图像尺寸分辩率不同的情况可以采用**降低采样或归一化方法**。

* **相似度匹配。**如存储的是**特征向量**，则比较特征向量之间的**加权**后的**平方距离**。如存储的是**散列码**，则比较**Hamming距离**。初筛后，还可以进一步筛选最佳图像集。

## 2. 图片搜索引擎算法及框架设计

### **基本步骤**

* 采用颜色空间特征提取器和构图空间特征提取器**提取图像特征**。
* 图像索引表构建驱动程序**生成待搜索图像库的图像特征索引表**。
* 图像搜索引擎驱动程序**执行搜索命令**，生成原图图像特征并传入图片搜索匹配器。
* 图片搜索匹配内核**执行搜索匹配任务**。返回前`limit`个最佳匹配图像。

### **所需模块**

* **numpy。**科学计算和矩阵运算利器。
* **cv2。**OpenCV的python模块接入。
* **re。**正则化模块。解析csv中的图像构图特征和色彩特征集。
* **csv。**高效地读入csv文件。
* **glob。**正则获取文件夹中文件路径。
* **argparse。**设置命令行参数。


### **封装类及驱动程序**

* **颜色空间特征提取器ColorDescriptor。**

1. **类成员`bins`。**记录HSV色彩空间生成的**色相、饱和度及明度**分布直方图的最佳bins分配。bins分配过多则可能导致程序效率低下，匹配难度和匹配要求过分苛严；bins分配过少则会导致匹配精度不足，不能表证图像特征。
2. **成员函数`getHistogram(self, image, mask, isCenter)`**。生成图像的色彩特征分布直方图。`image`为待处理图像，`mask`为图像处理区域的掩模，`isCenter`判断是否为图像中心，从而有效地对色彩特征向量做加权处理。权重`weight`取`5.0`。采用OpenCV的`calcHist()`方法获得直方图，`normalize()`方法归一化。
3. **成员函数`describe(self, image)`**。将图像从BGR色彩空间转为HSV色彩空间（此处应注意OpenCV读入图像的色彩空间为BGR而非RGB）。生成**左上、右上、左下、右下、中心部分的掩模**。中心部分掩模的形状为**椭圆形**。这样能够有效区分中心部分和边缘部分，从而在`getHistogram()`方法中对不同部位的色彩特征做**加权处理**。

```python
class ColorDescriptor:
	__slot__ = ["bins"]
	def __init__(self, bins):
		self.bins = bins
	def getHistogram(self, image, mask, isCenter):
		# get histogram
		imageHistogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		# normalize
		imageHistogram = cv2.normalize(imageHistogram, imageHistogram).flatten()
		if isCenter:
			weight = 5.0
			for index in xrange(len(imageHistogram)):
				imageHistogram[index] *= weight
		return imageHistogram
	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# get dimension and center
		height, width = image.shape[0], image.shape[1]
		centerX, centerY = int(width * 0.5), int(height * 0.5)
		# initialize mask dimension
		segments = [(0, centerX, 0, centerY), (0, centerX, centerY, height), (centerX, width, 0, centerY), (centerX, width, centerY, height)]
		# initialize center part
		axesX, axesY = int(width * 0.75) / 2, int (height * 0.75) / 2
		ellipseMask = numpy.zeros([height, width], dtype="uint8")
		cv2.ellipse(ellipseMask, (centerX, centerY), (axesX, axesY), 0, 0, 360, 255, -1)
		# initialize corner part
		for startX, endX, startY, endY in segments:
			cornerMask = numpy.zeros([height, width], dtype="uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipseMask)
			# get histogram of corner part
			imageHistogram = self.getHistogram(image, cornerMask, False)
			features.append(imageHistogram)
		# get histogram of center part
		imageHistogram = self.getHistogram(image, ellipseMask, True)
		features.append(imageHistogram)
		# return
		return features
```

* **构图空间特征提取器StructureDescriptor。**

1. **类成员`dimension`**。将所有图片归一化（降低采样）为`dimension`所规定的尺寸。由此才能够用于统一的匹配和构图空间特征的生成。
2. **成员函数`describe(self, image)`**。将图像从BGR色彩空间转为HSV色彩空间（此处应注意OpenCV读入图像的色彩空间为BGR而非RGB）。返回HSV色彩空间的矩阵，等待在搜索引擎核心中的下一步处理。

```python
class StructureDescriptor:
	__slot__ = ["dimension"]
	def __init__(self, dimension):
		self.dimension = dimension
	def describe(self, image):
		image = cv2.resize(image, self.dimension, interpolation=cv2.INTER_CUBIC)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		return image
```

* **图片搜索匹配内核Searcher。**

1. **类成员`colorIndexPath`和`structureIndexPath`**。记录色彩空间特征索引表路径和结构特征索引表路径。
2. **成员函数`solveColorDistance(self, features, queryFeatures, eps = 1e-5)`**。求`features`和`queryFeatures`特征向量的**二范数**。`eps`是为了**避免除零错误**。
3. **成员函数`solveStructureDistance(self, structures, queryStructures, eps = 1e-5)`**。同样是求特征向量的**二范数**。`eps`是为了**避免除零错误**。需作统一化处理，color和structure特征向量距离相对比例适中，不可过分偏颇。
4. **成员函数`searchByColor(self, queryFeatures)`**。使用csv模块的reader方法读入索引表数据。采用re的split方法解析数据格式。用字典`searchResults`存储query图像与库中图像的距离，键为图库内图像名`imageName`，值为距离`distance`。
5. **成员函数`transformRawQuery(self, rawQueryStructures)`**。将未处理的query图像矩阵转为用于匹配的**特征向量形式**。
6. **成员函数`searchByStructure(self, rawQueryStructures`)**。类似4。
7. **成员函数`search(self, queryFeatures, rawQueryStructures, limit = 3`)**。将`searchByColor`方法和`searchByStructure`的结果汇总，获得总匹配分值，分**值越低代表综合距离越小**，匹配程度越高。返回前`limit`个最佳匹配图像。

```python
class Searcher:
	__slot__ = ["colorIndexPath", "structureIndexPath"]
	def __init__(self, colorIndexPath, structureIndexPath):
		self.colorIndexPath, self.structureIndexPath = colorIndexPath, structureIndexPath
	def solveColorDistance(self, features, queryFeatures, eps = 1e-5):
		distance = 0.5 * numpy.sum([((a - b) ** 2) / (a + b + eps) for a, b in zip(features, queryFeatures)])
		return distance
	def solveStructureDistance(self, structures, queryStructures, eps = 1e-5):
		distance = 0
		normalizeRatio = 5e3
		for index in xrange(len(queryStructures)):
			for subIndex in xrange(len(queryStructures[index])):
				a = structures[index][subIndex]
				b = queryStructures[index][subIndex]
				distance += (a - b) ** 2 / (a + b + eps)
		return distance / normalizeRatio
	def searchByColor(self, queryFeatures):
		searchResults = {}
		with open(self.colorIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				features = []
				for feature in line[1:]:
					feature = feature.replace("[", "").replace("]", "")
					findStartPosition = 0
					feature = re.split("\s+", feature)
					rmlist = []
					for index, strValue in enumerate(feature):
						if strValue == "":
							rmlist.append(index)
					for _ in xrange(len(rmlist)):
						currentIndex = rmlist[-1]
						rmlist.pop()
						del feature[currentIndex]
					feature = [float(eachValue) for eachValue in feature]
					features.append(feature)
				distance = self.solveColorDistance(features, queryFeatures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "feature", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults
	def transformRawQuery(self, rawQueryStructures):
		queryStructures = []
		for substructure in rawQueryStructures:
			structure = []
			for line in substructure:
				for tripleColor in line:
					structure.append(float(tripleColor))
			queryStructures.append(structure)
		return queryStructures
	def searchByStructure(self, rawQueryStructures):
		searchResults = {}
		queryStructures = self.transformRawQuery(rawQueryStructures)
		with open(self.structureIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				structures = []
				for structure in line[1:]:
					structure = structure.replace("[", "").replace("]", "")
					structure = re.split("\s+", structure)
					if structure[0] == "":
						structure = structure[1:]
					structure = [float(eachValue) for eachValue in structure]
					structures.append(structure)
				distance = self.solveStructureDistance(structures, queryStructures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "structure", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults
	def search(self, queryFeatures, rawQueryStructures, limit = 3):
		featureResults = self.searchByColor(queryFeatures)
		structureResults = self.searchByStructure(rawQueryStructures)
		results = {}
		for key, value in featureResults.iteritems():
			results[key] = value + structureResults[key]
		results = sorted(results.iteritems(), key = lambda item: item[1], reverse = False)
		return results[ : limit]
```

* **图像索引表构建驱动index.py。**

* 引入`color_descriptor`和`structure_descriptor`。用于解析图片库图像，获得色彩空间特征向量和构图空间特征向量。
* 用`argparse`设置**命令行参数**。参数包括图片库路径、色彩空间特征索引表路径、构图空间特征索引表路径。
* 用`glob`获得图片库路径。
* 生成索引表文本并写入csv文件。
* 可采用如下命令行形式启动驱动程序。

```
python index.py --dataset dataset --colorindex color——index.csv --structure structure_index.csv
```

dataset为图片库路径。color\_index.csv为色彩空间特征索引表路径。structure\_index.csv为构图空间特征索引表路径。

```python
import color_descriptor
import structure_descriptor
import glob
import argparse
import cv2

searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
arguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
colorDesriptor = color_descriptor.ColorDescriptor(idealBins)

output = open(arguments["colorindex"], "w")

for imagePath in glob.glob(arguments["dataset"] + "/*.jpg"):
	imageName = imagePath[imagePath.rfind("/") + 1 : ]
	image = cv2.imread(imagePath)
	features = colorDesriptor.describe(image)
	# write features to file
	features = [str(feature).replace("\n", "") for feature in features]
	output.write("%s,%s\n" % (imageName, ",".join(features)))
# close index file
output.close()

idealDimension = (16, 16)
structureDescriptor = structure_descriptor.StructureDescriptor(idealDimension)

output = open(arguments["structureindex"], "w")

for imagePath in glob.glob("dataset" + "/*.jpg"):
	imageName = imagePath[imagePath.rfind("/") + 1 : ]
	image = cv2.imread(imagePath)
	structures = structureDescriptor.describe(image)
	# write structures to file
	structures = [str(structure).replace("\n", "") for structure in structures]
	output.write("%s,%s\n" % (imageName, ",".join(structures)))
# close index file
output.close()

```

* **图像搜索引擎驱动searchEngine.py。**

* 引入`color_descriptor`和`structure_descriptor`。用于解析待匹配（搜索）的图像，获得色彩空间特征向量和构图空间特征向量。
* 用`argparse`设置**命令行参数**。参数包括图片库路径、色彩空间特征索引表路径、构图空间特征索引表路径、待搜索图片路径。
* 生成索引表文本并写入csv文件。
* 可采用如下命令行形式启动驱动程序。

```
python searchEngine.py -c color_index.csv -s structure_index.csv -r dataset -q query/pyramid.jpg 
```

dataset为图片库路径。color\_index.csv为色彩空间特征索引表路径。structure\_index.csv为构图空间特征索引表路径，query/pyramid.jpg为待搜索图片路径。


```python
searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
searchArgParser.add_argument("-q", "--query", required = True, help = "Path to the query image")
searchArgParser.add_argument("-r", "--resultpath", required = True, help = "Path to the result path")
searchArguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
idealDimension = (16, 16)

colorDescriptor = color_descriptor.ColorDescriptor(idealBins)
structureDescriptor = structure_descriptor.StructureDescriptor(idealDimension)
queryImage = cv2.imread(searchArguments["query"])
colorIndexPath = searchArguments["colorindex"]
structureIndexPath = searchArguments["structureindex"]
resultPath = searchArguments["resultpath"]

queryFeatures = colorDescriptor.describe(queryImage)
queryStructures = structureDescriptor.describe(queryImage)

imageSearcher = searcher.Searcher(colorIndexPath, structureIndexPath)
searchResults = imageSearcher.search(queryFeatures, queryStructures)

for imageName, score in searchResults:
	queryResult = cv2.imread(resultPath + "/" + imageName)
	cv2.imshow("Result Score: " + str(int(score)) + " (lower is better)", queryResult)
	cv2.waitKey(0)

cv2.imshow("Query", queryImage)
cv2.waitKey(0)
```


## 3. 搜索引擎测试
###Qeury: fish.jpg

![fish](http://img.blog.csdn.net/20150620211414923)

###Result(匹配分值越低越好):
1) Score: 0

![fish](http://img.blog.csdn.net/20150620211414923)

2) Score: 17

![fish](http://img.blog.csdn.net/20150620211546787)

3) Score: 21

![fish](http://img.blog.csdn.net/20150620211643281)

###Qeury: forest.jpg

![forest](http://img.blog.csdn.net/20150620211820008)

###Result(匹配分值越低越好):
1) Score: 0

![forest](http://img.blog.csdn.net/20150620211820008)

2) Score: 33

![forest](http://img.blog.csdn.net/20150620211914110)

3) Score: 33

![forest](http://img.blog.csdn.net/20150620212007933)

###Qeury: trip.jpg

![trip](http://img.blog.csdn.net/20150620212152590)

###Result(匹配分值越低越好):
1) Score: 0

![trip](http://img.blog.csdn.net/20150620212152590)

2) Score: 23

![trip](http://img.blog.csdn.net/20150620212241244)

3) Score: 24

![trip](http://img.blog.csdn.net/20150620212313457)

###Qeury: zebra.jpg

![zebra](http://img.blog.csdn.net/20150620212348223)

###Result(匹配分值越低越好):
1) Score: 0

![zebra](http://img.blog.csdn.net/20150620212348223)

2) Score: 23

![zebra](http://img.blog.csdn.net/20150620212440638)

3) Score: 25

![zebra](http://img.blog.csdn.net/20150620212507828)

**总结：**总能搜索到完全一致的图像（即原图）。搜索得到的图像与原图基本符合。测试成功。如分析有误或代码出错，请批评指正。谢谢。

## 4.  Python源代码
###`color_descriptor.py`

```python
import cv2
import numpy

class ColorDescriptor:
	__slot__ = ["bins"]
	def __init__(self, bins):
		self.bins = bins
	def getHistogram(self, image, mask, isCenter):
		# get histogram
		imageHistogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		# normalize
		imageHistogram = cv2.normalize(imageHistogram, imageHistogram).flatten()
		if isCenter:
			weight = 5.0
			for index in xrange(len(imageHistogram)):
				imageHistogram[index] *= weight
		return imageHistogram
	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# get dimension and center
		height, width = image.shape[0], image.shape[1]
		centerX, centerY = int(width * 0.5), int(height * 0.5)
		# initialize mask dimension
		segments = [(0, centerX, 0, centerY), (0, centerX, centerY, height), (centerX, width, 0, centerY), (centerX, width, centerY, height)]
		# initialize center part
		axesX, axesY = int(width * 0.75) / 2, int (height * 0.75) / 2
		ellipseMask = numpy.zeros([height, width], dtype="uint8")
		cv2.ellipse(ellipseMask, (centerX, centerY), (axesX, axesY), 0, 0, 360, 255, -1)
		# initialize corner part
		for startX, endX, startY, endY in segments:
			cornerMask = numpy.zeros([height, width], dtype="uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipseMask)
			# get histogram of corner part
			imageHistogram = self.getHistogram(image, cornerMask, False)
			features.append(imageHistogram)
		# get histogram of center part
		imageHistogram = self.getHistogram(image, ellipseMask, True)
		features.append(imageHistogram)
		# return
		return features
```

###`structure_descriptor.py`

```python
import cv2

class StructureDescriptor:
	__slot__ = ["dimension"]
	def __init__(self, dimension):
		self.dimension = dimension
	def describe(self, image):
		image = cv2.resize(image, self.dimension, interpolation=cv2.INTER_CUBIC)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		return image

```

###`searcher.py`

```python
import numpy
import csv
import re

class Searcher:
	__slot__ = ["colorIndexPath", "structureIndexPath"]
	def __init__(self, colorIndexPath, structureIndexPath):
		self.colorIndexPath, self.structureIndexPath = colorIndexPath, structureIndexPath
	def solveColorDistance(self, features, queryFeatures, eps = 1e-5):
		distance = 0.5 * numpy.sum([((a - b) ** 2) / (a + b + eps) for a, b in zip(features, queryFeatures)])
		return distance
	def solveStructureDistance(self, structures, queryStructures, eps = 1e-5):
		distance = 0
		normalizeRatio = 5e3
		for index in xrange(len(queryStructures)):
			for subIndex in xrange(len(queryStructures[index])):
				a = structures[index][subIndex]
				b = queryStructures[index][subIndex]
				distance += (a - b) ** 2 / (a + b + eps)
		return distance / normalizeRatio
	def searchByColor(self, queryFeatures):
		searchResults = {}
		with open(self.colorIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				features = []
				for feature in line[1:]:
					feature = feature.replace("[", "").replace("]", "")
					findStartPosition = 0
					feature = re.split("\s+", feature)
					rmlist = []
					for index, strValue in enumerate(feature):
						if strValue == "":
							rmlist.append(index)
					for _ in xrange(len(rmlist)):
						currentIndex = rmlist[-1]
						rmlist.pop()
						del feature[currentIndex]
					feature = [float(eachValue) for eachValue in feature]
					features.append(feature)
				distance = self.solveColorDistance(features, queryFeatures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "feature", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults
	def transformRawQuery(self, rawQueryStructures):
		queryStructures = []
		for substructure in rawQueryStructures:
			structure = []
			for line in substructure:
				for tripleColor in line:
					structure.append(float(tripleColor))
			queryStructures.append(structure)
		return queryStructures
	def searchByStructure(self, rawQueryStructures):
		searchResults = {}
		queryStructures = self.transformRawQuery(rawQueryStructures)
		with open(self.structureIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				structures = []
				for structure in line[1:]:
					structure = structure.replace("[", "").replace("]", "")
					structure = re.split("\s+", structure)
					if structure[0] == "":
						structure = structure[1:]
					structure = [float(eachValue) for eachValue in structure]
					structures.append(structure)
				distance = self.solveStructureDistance(structures, queryStructures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "structure", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults
	def search(self, queryFeatures, rawQueryStructures, limit = 3):
		featureResults = self.searchByColor(queryFeatures)
		structureResults = self.searchByStructure(rawQueryStructures)
		results = {}
		for key, value in featureResults.iteritems():
			results[key] = value + structureResults[key]
		results = sorted(results.iteritems(), key = lambda item: item[1], reverse = False)
		return results[ : limit]

```

###`index.py`

```python
import color_descriptor
import structure_descriptor
import glob
import argparse
import cv2

searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
arguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
colorDesriptor = color_descriptor.ColorDescriptor(idealBins)

output = open(arguments["colorindex"], "w")

for imagePath in glob.glob(arguments["dataset"] + "/*.jpg"):
	imageName = imagePath[imagePath.rfind("/") + 1 : ]
	image = cv2.imread(imagePath)
	features = colorDesriptor.describe(image)
	# write features to file
	features = [str(feature).replace("\n", "") for feature in features]
	output.write("%s,%s\n" % (imageName, ",".join(features)))
# close index file
output.close()

idealDimension = (16, 16)
structureDescriptor = structure_descriptor.StructureDescriptor(idealDimension)

output = open(arguments["structureindex"], "w")

for imagePath in glob.glob("dataset" + "/*.jpg"):
	imageName = imagePath[imagePath.rfind("/") + 1 : ]
	image = cv2.imread(imagePath)
	structures = structureDescriptor.describe(image)
	# write structures to file
	structures = [str(structure).replace("\n", "") for structure in structures]
	output.write("%s,%s\n" % (imageName, ",".join(structures)))
# close index file
output.close()

```

###`searchEngine.py`

```python
import color_descriptor
import structure_descriptor
import searcher
import argparse
import cv2

searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
searchArgParser.add_argument("-q", "--query", required = True, help = "Path to the query image")
searchArgParser.add_argument("-r", "--resultpath", required = True, help = "Path to the result path")
searchArguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
idealDimension = (16, 16)

colorDescriptor = color_descriptor.ColorDescriptor(idealBins)
structureDescriptor = structure_descriptor.StructureDescriptor(idealDimension)
queryImage = cv2.imread(searchArguments["query"])
colorIndexPath = searchArguments["colorindex"]
structureIndexPath = searchArguments["structureindex"]
resultPath = searchArguments["resultpath"]

queryFeatures = colorDescriptor.describe(queryImage)
queryStructures = structureDescriptor.describe(queryImage)

imageSearcher = searcher.Searcher(colorIndexPath, structureIndexPath)
searchResults = imageSearcher.search(queryFeatures, queryStructures)

for imageName, score in searchResults:
	queryResult = cv2.imread(resultPath + "/" + imageName)
	cv2.imshow("Result Score: " + str(int(score)) + " (lower is better)", queryResult)
	cv2.waitKey(0)

cv2.imshow("Query", queryImage)
cv2.waitKey(0)

```

###`searchEngineTest.py`

```python
import cv2
import glob
import csv
import re
import numpy
import structure_descriptor

idealDimension = (16, 16)
structureDescriptor = structure_descriptor.StructureDescriptor(idealDimension)

testImage = cv2.imread("query/forest.jpg")
rawQueryStructures = structureDescriptor.describe(testImage)

# index
output = open("structureIndex.csv", "w")

for imagePath in glob.glob("dataset" + "/*.jpg"):
	imageName = imagePath[imagePath.rfind("/") + 1 : ]
	image = cv2.imread(imagePath)
	structures = structureDescriptor.describe(image)
	# write structures to file
	structures = [str(structure).replace("\n", "") for structure in structures]
	output.write("%s,%s\n" % (imageName, ",".join(structures)))
# close index file
output.close()

# searcher

def solveStructureDistance(self, structures, queryStructures, eps = 1e-5):
	distance = 0
	for index in xrange(len(queryFeatures)):
		for subIndex in xrange(len(queryFeatures[index])):
			a = features[index][subIndex]
			b = queryFeatures[index][subIndex]
			distance += (a - b) ** 2 / (a + b + eps)
	return distance / 5e3

queryStructures = []
for substructure in rawQueryStructures:
	structure = []
	for line in substructure:
		for tripleColor in line:
			structure.append(float(tripleColor))
	queryStructures.append(structure)
searchResults = {}
with open("structureIndex.csv") as indexFile:
	reader = csv.reader(indexFile)
	for line in reader:
		structures = []
		for structure in line[1:]:
			structure = structure.replace("[", "").replace("]", "")
			structure = re.split("\s+", structure)
			if structure[0] == "":
				structure = structure[1:]
			structure = [float(eachValue) for eachValue in structure]
			print len(structure)
			structures.append(structure)
		distance = solveDistance(structures, queryStructures)
		searchResults[line[0]] = distance
	indexFile.close()
searchResults = sorted(searchResults.iteritems(), key=lambda item: item[1], reverse=False)

print searchResults

```

