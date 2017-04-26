# This implements Emotion Recognition using
# "Recognition of Facial Expressions and Measurement of Levels of Interest From Video"
# by M.Yeasin, B.Bullot and R.Sharma.

import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.externals import joblib
from hmmlearn import hmm

out = open ('output', 'w')

# Train, test, save and load model 
class EmotionRecognizer:

	def __init__ (self, pneighbors_knn, pn_components_hmm):
		# model parameters
		self.neighbors_knn = pneighbors_knn
		self.weights_knn = 'uniform'
		self.n_components_hmm = pn_components_hmm

		# models
		self.knn = None
		self.hmms = None
		
		self.emotions = {'Angry':0, 'Neutral':1, 'Shocked':2, 'Smiling':3}
		self.emotionsr = ['Angry', 'Neutral', 'Shocked', 'Smiling']
		self.nemotions = 4

		self.face_cascade = cv2.CascadeClassifier ('data/haarcascade_frontalface_default.xml')

	# From a video this will calculate optical flow
	# parameters : filePath -> path of file to calculate optical flow
	# Returns : 2d numpy array of optical flows each 1D array in that 2d array will 
	#           have x and y components of optical flows side by side

	def calculateFlow (self, filePath):
		cap = cv2.VideoCapture (filePath)
		if cap is None:
			print ('File not found')
			return
		else:
			print ('reading {0}'.format (filePath))

		flows = []
		grayFramep = None

		while True:
			ret, frame = cap.read ()
			if not ret: 
				print ('reading finished')
				break

			grayFramep = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)

			faces = self.face_cascade.detectMultiScale (grayFramep, 1.3, 5)

			if len (faces) is 1:
				x,y,w,h = faces[0]
				# cv2.imshow ('image', grayFramep[y:y+h, x:x+w])
				# cv2.waitKey(1000)
				grayFramep = cv2.resize (grayFramep[y:y+h, x:x+w], (40, 40), interpolation = cv2.INTER_CUBIC)
				break
			# else:
				# print ('face not detected')

		while True:
			grayFrame = None
			ret = True

			while True:
				ret, frame = cap.read ()
			
				if not ret:	
					# print ('reading finished ..')
					break

				grayFrame = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
				faces = self.face_cascade.detectMultiScale (grayFrame, 1.3, 5)
				
				if len(faces) is 1:
					x,y,w,h = faces[0]
					# cv2.imshow ('image', grayFrame[y:y+h, x:x+w])
					# k = cv2.waitKey(100) & 0xff
					grayFrame = cv2.resize (grayFrame[y:y+h, x:x+w], (40, 40), interpolation = cv2.INTER_CUBIC)
					break
				# else:
					# print ('face not detected ..')
			
			if not ret:	break

			flow = cv2.calcOpticalFlowFarneback (grayFramep, grayFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			flowx = np.reshape (flow[...,0], (1600))
			flowy = np.reshape (flow[...,1], (1600))
			flowxy = np.concatenate ((flowx, flowy), axis = 0)

			flows.append (flowxy)
			grayFramep = grayFrame

		flows = np.array (flows)

		cap.release ()
		print ('reading {0} finished'.format (filePath))
		return flows

	# This calculate optical flow of video and reduce dimensions of optical flow using
	# singular vector decomposition and returns list of reduced optical flow and emotion for that video
	# Parameters : fileName -> path of video to encode
	# Returns    : list of optical flows of images in video after dimensionality reduction using PCA
	def encodeVideo (self, fileName):
		y = None
		# if False:
		if os.path.exists (fileName+'.npa'):
			print ('loading {0}'.format (fileName+'npa'))
			y = np.load (fileName+'.npa')
		else:
			flows = self.calculateFlow (fileName)

			if flows.shape[0] == 0: return None

			x_std = StandardScaler().fit_transform (flows)

			# v is an array of eigen vectors and s is an array of eigen values
			u, s, v = np.linalg.svd (x_std)

			eig_vecs = v[:,0:40]
			# making transformation using eigen vectors by matrix multiplication
			y = x_std.dot(eig_vecs)
			y.dump (fileName+'.npa')

		i = 0
		cflows = []
		for flow in y:
			if i%1 == 0:
				cflows.append (flow)
			i += 1

		return cflows

	# generates temporal sequences for all videos
	# Parameters : trainData -> a list of tuples with file name and emotion
	# Returns    : numpy array of temporal sequences of all videos combined,
	#              list of sizes of videos and
	#              list of emotions of videos.
	def genTemporalSequence (self, trainData):
		# list of flows for each image flow in all videos
		allFlows = []
		# list of tags for each image flow in all videos
		tags = []
		blockSizes = []
		blockTags = []

		# i = 0
		for fileName, tagname in trainData:
			# if i > 4: break

			flows = self.encodeVideo (fileName)
			if flows is None: continue
			allFlows += flows
			tags += [tagname]*len(flows)

			blockSizes.append (len(flows))
			blockTags.append (tagname)

			# i += 1

		allFlows = np.array (allFlows)
		tags = np.array (tags)

		# train KNN using optical flows and their emotions
		self.knn = neighbors.KNeighborsClassifier (self.neighbors_knn, weights=self.weights_knn)
		self.knn.fit (allFlows, tags)

		# test "training data" and generate a temporal sequence
		temporalSequence = self.knn.predict (allFlows)

		print ('temporal sequence shape ', temporalSequence.shape)

		print (allFlows.shape)
		print (blockSizes)
		print (blockTags)

		print ('finished generating sequence')

		return (temporalSequence, blockSizes, blockTags)

	# Train KNN and HMM and store it in class variable
	# Parameters : dataDir -> a list of tuples with file name and emotion
	def train (self, dataDir):
		temporalSequence, blockSizes, blockTags = self.genTemporalSequence (dataDir)
		train = [[] for i in range (self.nemotions)]
		trainblocks = [[] for i in range (self.nemotions)]
		pcum = 0
		cum = 0

		i = 0
		print (temporalSequence, blockSizes, blockTags)
		for size in blockSizes:
			emotion = blockTags [i]
			cum = cum + size
			train[emotion].append (temporalSequence[pcum:cum])

			trainblocks[emotion].append (size)
			pcum = cum
			i += 1

		genTemporalSequence = []

		i = 0
		while i < self.nemotions:
			print (train[i])
			train[i] = np.hstack (train[i])
			train[i] = train[i][:,None]
			trainblocks[i] = np.array (trainblocks[i])
			i += 1

		self.hmms = [[] for i in range (self.nemotions)]

		for i in range (self.nemotions):
			self.hmms[i] = hmm.GaussianHMM (n_components=self.n_components_hmm, covariance_type="full", n_iter=100)
			self.hmms[i].fit (train[i], trainblocks[i])

	# Tests emotion of a single file using trained KNN and HMMs
	# Parameters : fileName -> path of file to check emotion
	#              emotion  -> actual emotion of that video 
	# Returns    : whether actual emotion matched to predicted emotion or not
	def testFile (self, fileName, emotion):
		print ('testing {0}'.format (fileName))
		flows = self.encodeVideo (fileName)
		flows = np.array (flows)

		if flows is None: return

		tempseq = self.knn.predict (flows)[:,None]
		mxProb = -1000000
		mxProbIdx = -1
		mxProb1 = -1000000
		mxProbIdx1 = -1

		for i in range (self.nemotions):
			x = self.hmms[i].score (tempseq)
			# print (x)
			if x>mxProb:
				mxProb1 = mxProb
				mxProbIdx1 = mxProbIdx
				mxProb = x
				mxProbIdx = i
			elif x>mxProb1:
				mxProb1 = x
				mxProbIdx1 = i

		out.write ('Emotion {0} with log likelyhood {1}, actual is {2}\n'. \
			format (self.emotionsr[mxProbIdx], mxProb, self.emotionsr[emotion]))
		out.write ('Emotion {0} with log likelyhood {1}, actual is {2}\n\n'. \
			format (self.emotionsr[mxProbIdx1], mxProb1, self.emotionsr[emotion]))

		if emotion is mxProbIdx:
			return True
		else:
			return False

	# Does testing of a data in a directory
	# Parameters : testData -> a list of tuples with file name and emotion
	# Returns : accuracy of data
	def test (self, testData):
		total = 0
		correct = 0
		# i = 0
		for fileName, emotion in testData:
			# if i >= 5: break
			# i += 1
			x = self.testFile (fileName, emotion)
			if x is None: continue
			if x:
				correct += 1
			total += 1

		return correct/total

	# saves models into disk
	def dump (self, modelDir):
		try:
			joblib.dump (self.knn, modelDir + '/knn.dump')
			for key, value in self.emotions.items ():
				joblib.dump (self.hmms[value], modelDir + '/hmm'+key+'.dump')
			print ('dumped models to {0}'.format (modelDir))
		except ZeroDivisionError as e:
			print ('Error in dumping models')
			return False

		return True

	# loads models from disk
	def load (self, modelDir):
		try:
			self.knn = joblib.load (modelDir + '/knn.dump')
			self.hmms = [[] for i in range(self.nemotions)]
			for key, value in self.emotions.items ():
				self.hmms[value] = joblib.load (modelDir + '/hmm'+key+'.dump')
			print ('loaded models from {0}'.format (modelDir))
		except Exception as e:
			print ('Error in loading models')
			return False

			return True

############################## Main #######################################

ndata = 20
tndata = 5
gneighbors_knn = 3000
gn_components_hmm = 8

# data organization should be, a folder of all subjects
# in the subject folder, folders should be there corresponds to each emotion
# the emotion folder should contain videos
def dataGen ():
	src = 'data/datasets/PerUser-Emotion-Classifier/Subjects'
	emotions = {'Angry':0, 'Neutral':1, 'Shocked':2, 'Smiling':3}
	testData = []
	trainData = []

	for subject in  os.listdir (src):
		for subemotion in os.listdir(src+'/'+subject):
			x = subemotion.split('-')
			if x[0] == 'Test':
				i = 0
				for file in os.listdir (src+'/'+subject+'/'+subemotion):
					if file.endswith ('.npa'):
						continue
					if i >= tndata: break
					i += 1
					testData.append ((src+'/'+subject+'/'+subemotion+'/'+file, emotions[x[1]]))
			else:
				i = 0
				for file in os.listdir (src+'/'+subject+'/'+subemotion):
					if file.endswith ('.npa'):
						continue
					if i >= ndata: break
					i += 1
					trainData.append ((src+'/'+subject+'/'+subemotion+'/'+file, emotions[x[0]]))

	return (trainData, testData)

def main ():
	trainData, testData = dataGen ()
	directory = 'data/model_'+str(gneighbors_knn)+'_'+str(gn_components_hmm)+'_'+str(ndata)+'_'+str(tndata)
	
	
	er = EmotionRecognizer (gneighbors_knn, gn_components_hmm)
	# er.train (trainData)

	# if not os.path.exists(directory):
	# 	os.makedirs(directory)
	# er.dump (directory)

	er.load (directory)
	acc = er.test (testData)

	out.write ('accuracy is {0}\n'.format (acc))

	out.close ()
	
main ()

