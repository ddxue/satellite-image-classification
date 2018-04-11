import numpy as np
import random
import sys     # imports the sys module
import collections
import math
from collections import Counter

############################################################
# k-means
############################################################

def kmeans(examples, K, maxIters):
	'''
	examples: list of examples, each example is a int-to-int dict representing a sparse vector.
	K: number of desired clusters. Assume that 0 < K <= |examples|.
	maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
	Return: (length K list of cluster centroids,
			list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
			final reconstruction loss)
	'''
	# Caches
	exampleSize_cache = [None] * examples.shape[0]
	centerSize_cache = [None] * K
	
	# Critical Variables
	n = len(examples)                           # Number of Training Points/Examples
	centers = random.sample(examples, K)        # Cluster centroids
	assignments = [None] * n                    # Cluster assignments

	def dist(examples, centers, example_index, center_index):
		example = examples[example_index]
		center = centers[center_index]

		if exampleSize_cache[example_index]:
			example_size = exampleSize_cache[example_index]
		else:
			example_size = dotProduct(example,example)
			exampleSize_cache[example_index] = example_size

		if centerSize_cache[center_index]:
			center_size = centerSize_cache[center_index]
		else:
			center_size = dotProduct(center,center)
			centerSize_cache[center_index] = center_size

		return example_size + center_size - 2.0*dotProduct(example,center)

	def calculateCenter(k):
		exampleCluster = [examples[i] for i in range(n) if assignments[i] == k]
		clusterSize = len(exampleCluster)
		if clusterSize > 0:
			new_center = {}
			# Find the sum mapping.
			for example in exampleCluster:
				for key,val in example.items():
					if key in new_center:
						new_center[key] += val
					else:
						new_center[key] = val
			
			# Take the average.
			for ckey,cval in new_center.items():
				new_center[ckey] = 1.0 * cval / clusterSize
			
			# Update Center.
			if centers[k] != new_center: # new center
				centers[k] = new_center
				centerSize_cache[k] = None

	# Algorithm Main Loop.
	for iteration in range(maxIters):
		totalCost = 0
		new_assignments = [None] * n
		
		# Step 1: set assignments (Assign example i to cluster with closest centroid).
		for i in range(n):
			cost, new_assignments[i] = min([(dist(examples, centers, i, k), k) for k in range(K)])
			totalCost += cost
		# Terminate Early if Converges.
		if new_assignments == assignments:
			return centers, assignments, totalCost
		else:
			assignments = new_assignments
		
		# Step 2: set centers (average of examples assignned to cluster k).
		for k in range(K):
			calculateCenter(k)

	return centers, assignments

def dotProduct(d1, d2):
	"""
	@param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
	@param dict d2: same as d1
	@return float: the dot product between d1 and d2
	"""
	if len(d1) < len(d2):
		return dotProduct(d2, d1)
	else:
		return np.dot(d1, d2)

# def extractPixelFeatures(pixels):
#     """
#     Extract pixel features from a list of pixels.
#     @param string x: 
#     @return dict: feature vector representation of x.
#     Example: "[255,255,122,...]" --> {0:255,1:255,2:122,...}
#     """
#     featureDict = {}
#     for i in range(len(pixels)):
#         featureDict[i] = pixels[i]
#     return featureDict


############################################################
# Stochastic Gradient Descent (SGD)
############################################################

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
	'''
	Given |trainExamples| and |testExamples| (each one is a list of (x,y)
	pairs), a |featureExtractor| to apply to x, and the number of iterations to
	train |numIters|, the step size |eta|, return the weight vector (sparse
	feature vector) learned.

	You should implement stochastic gradient descent.

	Note: only use the trainExamples for training!
	You should call evaluatePredictor() on both trainExamples and testExamples
	to see how you're doing as you learn after each iteration.
	'''
	weights = {}  # feature => weight
	def gradient(weights,feature,y):
		if len(feature) == 0:
			return 0
		if dotProduct(weights,feature)*y < 1:
			val = {}
			for k,v in feature.items():
				val[k] = v*y*(-1)
			return val
		else:
			return 0

	for i in range(numIters):
		for x,y in trainExamples:
			feature = featureExtractor(x)
		  
			loss = gradient(weights,feature,y)
			if loss is not 0:
				increment(weights,(-1)*eta,loss)
		#evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))

	return weights
#learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)


# ################################################
# ################ RUN OUR SCRIPT ################
# ################################################

# # Each data set is a list of (pixels, emotion) tuples
# training_data, testing_data = parseKaggleData("fer2013.csv")




# ###### Build Model Phase #####

# # Build up the training examples
# trainingExamples = []
# for picture in training_data:
#   pixels, emotion = picture
#   pictureFeatures = extractPixelFeatures(pixels)
#   trainingExamples.append(pictureFeatures)

# print "Done loading examples." 

# # Caches
# exampleSize_cache = [None] * len(trainingExamples) 
# centerSize_cache = [None] * NUM_EMOTIONS

# # Run K-Means on Training Data
# # - centers is a list of K feature vectors
# # - assignments is a mapping of each training example to a center
# centers, assignments = kmeans(trainingExamples, NUM_EMOTIONS, 5)

# print "Done training k-means on examples." 

# # Find most common emotion for each centroid
# centerEmotions = [collections.defaultdict(int, {}) for i in centers]
# for i, assignment in enumerate(assignments):
#   emotion = training_data[i][1]
#   centerEmotions[assignment][emotion] += 1

# centerClassifications = {}
# for center in range(len(centerEmotions)):
#   centerClassifications[center] = max(centerEmotions[center], key=centerEmotions[center].get)




# ###### Evaluation Phase #####

# # Build up the testing examples
# testingExamples = []
# testingEmotions = []
# for picture in testing_data:
#   pixels, emotion = picture
#   pictureFeatures = extractPixelFeatures(pixels)
#   testingExamples.append(pictureFeatures)
#   testingEmotions.append(emotion)

# print "Done loading testing examples." 

# exampleSize_cache = [None] * len(testingExamples) 
# centerSize_cache = [None] * NUM_EMOTIONS

# # Calculate Result
# correctEmotion = 0.0
# incorrectEmotion = 0.0
# for i, example in enumerate(testingExamples):
#   print i
#   distances = {c: dist(testingExamples, centers, i, c) for c, center in enumerate(centers)}
#   assignment = min(distances, key=distances.get)
#   emotion = centerClassifications[assignment]
#   if emotion == testingEmotions[i]:
#       correctEmotion += 1
#   else:
#       incorrectEmotion += 1





# print "Correct: {}, Incorrect: {}".format(correctEmotion, incorrectEmotion)
# print (correctEmotion / (correctEmotion + incorrectEmotion)) 