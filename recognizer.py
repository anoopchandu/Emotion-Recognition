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

class EmotionRecognizer:

	def __init__ (self):
		# model parameters
		self.neighbors_knn = 10
		self.weights_knn = 'uniform'
		self.n_components_hmm = 40

		# models
		self.knn = None
		self.hmms = None
		
		self.emotions = {'surprise':0, 'happy':1, 'sad':2, 'anger':3, 'fear':4, 'disgust':5}
		self.emotionsr = ['surprise', 'happy','sad', 'anger', 'fear', 'disgust']

		self.face_cascade = cv2.CascadeClassifier ('data/haarcascade_frontalface_default.xml')

	# From a video this will calculate optical flow
	# parameters : filePath -> path of file to calculate optical flow
	# Returns : 2d numpy array of optical flows each 1D array in that 2d array will 
	#           have x and y components of optical flows side by side

	def calculateFlow (self, filePath):
		cap = cv2.VideoCapture (filePath)

		if cap == None:
			print ('File not found')
			return

		ret, frame = cap.read ()

		if not ret:
			cap.release ()
			print ('Unable to read video, exiting')
			return
		else:
			print ('reading ' + filePath)

		grayFramep = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale (grayFramep, 1.3, 5)

		if faces.shape[0] is not 1:
			cap.release ()
			print ('no faces or more than one face detected, exiting')
			return
		else:
			x,y,w,h = faces[0]

		grayFramep = cv2.resize (grayFramep[y:y+h, x:x+w], (40, 40), interpolation = cv2.INTER_CUBIC)

		flows = []

		while True:
			ret, frame = cap.read ()

			if not ret:
				print ('reading finished')
				break

			grayFrame = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale (grayFrame, 1.3, 5)

			if nfaces.shape[0] is not 1:
				print ('no faces or more than one face detected, exiting')
				return
			else:
				x,y,w,h = faces[0]

			grayFrame = cv2.resize (grayFrame[y:y+h, x:x+w], (40, 40), interpolation = cv2.INTER_CUBIC)

			flow = cv2.calcOpticalFlowFarneback (grayFramep, grayFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			flowx = np.reshape (flow[...,0], (1600))
			flowy = np.reshape (flow[...,1], (1600))
			flowxy = np.concatenate ((flowx, flowy), axis = 0)

			flows.append (flowxy)

		flows = np.array (flows)
		cap.release()
		return flows


	# This calculate optical flow of video and reduce dimensions of optical flow using
	# singular vector decomposition and returns list of reduced optical flow and emotion for that video
	# Parameters : fileName -> path of video to encode
	# Returns    : list of optical flows of images in video and
	#              emotion there in video
	def encodeVideo (self, fileName):
		tagname = None
		try:
			tagname = emotion [fileName.split ('_')[1]]
		except 	KeyError as ke:
			print ('no emotion for file')
			return (None,-1)

		flows = self.calculateFlow (fileName)

		x_std = StandardScaler().fit_transform (flows)

		# v is an array of eigen vectors and s is an array of eigen values
		u, s, v = np.linalg.svd (x_std)

		eig_vecs = v[:,0:40]
		# making transformation using eigen vectors by matrix multiplication
		y = x_std.dot(eig_vecs)

		i = 0
		cflows = []
		for flow in y:
			if i%5 == 0:
				cflows.append (flow)

		return (cflows, tagname)

	# generates temporal sequences for all videos
	# Parameters : dataDir -> directory of training data
	# Returns    : numpy array of temporal sequences of all videos combined,
	#              list of sizes of videos and
	#              list of emotions of videos.
	def genTemporalSequence (self, dataDir):
		# list of flows for each image flow in all videos
		allFlows = []
		# list of tags for each image flow in all videos
		tags = []
		blockSizes = []
		blockTags = []

		for fileName in os.listdir (dataDir):
			flows, tagname = self.encodeVideo (dataDir + '/' + fileName)
			
			if tagname is -1: continue

			allFlows += flows
			tags += [tagname]*len(flows)

			blockSizes.append (len(flows))
			blockTags.append (tagname)

		allFlows = np.array (allFlows)
		tags = np.array (tags)

		# train KNN using optical flows and their emotions
		self.knn = neighbors.KNeighborsClassifier(self.neighbors_knn, weights=self.weights_knn)
		self.knn.fit (allFlows, tags)

		# test "training data" and generate a temporal sequence
		temporalSequence = self.knn.predict (allFlows)

		return (temporalSequence, blockSizes, blockTags)

	# Train KNN and HMM and store it in class variable
	def train (self, dataDir):
		temporalSequence, blockSizes, blockTags = self.genTemporalSequence (dataDir)
		train = [[],[],[],[],[],[]]
		trainblocks = [[],[],[],[],[],[]]
		pcum = 0
		cum = 0
		for emotion, sizes in blockTags, blockSizes:
			cum = cum + size
			train[emotion]  += temporalSequence[pcum:cum-1]
			trainblocks[emotion].append (size)
			pcum = cum

		genTemporalSequence = []

		self.hmms = [None, None, None, None, None, None]

		for i in range(0,6):
			train[i] = np.array (train[i])[:,None]
			trainblocks = np.array (trainblocks[i])[:,None]
			self.hmms[i] = hmm.GaussianHMM(n_components=n_components_hmm, covariance_type="full", n_iter=100)
			self.hmms[i].fit (train[i], trainblocks[i])

	# Tests emotion of a single file using trained KNN and HMMs
	# and returns whether true emotions is matched or not 
	def testFile (self, fileName):
		flows, tagname = self.encodeVideo (fileName)
		flows = np.array (flows)
		tempseq = self.knn.predict (flows)
		mxProb = 0
		mxProbIdx = -1
		mxProb1 = 0
		mxProbIdx1 = -1

		for i in range(0,6):
			x = self.hmms[i].score (tempseq)
			if x>mxProb:
				mxProb1 = mxProb
				mxProbIdx1 = mxProbIdx
				mxProb = x
				mxProbIdx = i
			elif x>mxProb1:
				mxProb1 = x
				mxProbIdx1 = i

		print ('Emotion {0} with probability {1}', emotionsr[mxProbIdx], mxProb)
		print ('Emotion {0} with probability {1}', emotionsr[mxProbIdx1], mxProb1)

		if tagname is mxProbIdx:
			return True
		else
			return False

	# Does testing of a data in a directory
	# Parameters : dataDir -> directory of testing videos
	# Returs : accuracy of data
	def test (self, dataDir):
		if not os.path.isdir (dataDir):
			return -1

		total = 0
		correct = 0
		for file in os.listdir (dataDir):
			if os.path.isfile (dataDir+'/'+file):
				if self.testFile (dataDir+'/'+file):
					correct = correct + 1
				total = total + 1

		return correct/total

	# saves models into disk
	def dump (self, modelDir):
		try:
			joblib.dump (self.knn, modelDir + '/knn.dump')
			for key, value in self.emotions
				joblib.dump (self.hmm[value], modelDir + '/hmm'+key+'.dump')
			print ('dumped models to {0}', modelDir)
		except Exception as e:
			print ('Error in dumping models')
			return False

		return True

	# loads models from disk
	def load (self, modelDir):
		try:
			self.knn = joblib.load (modelDir + '/knn.dump')
			for key, value in self.emotions
				self.hmm[value] = joblib.load (modelDir + '/hmm'+key+'.dump')
			print ('loaded models from {0}', modelDir)
		except Exception as e:
			print ('Error in loading models')
			return False

		return True

def start ():
	er = EmotionRecognizer ()
	er.train ('data/train')
	er.dump ('data/model')
	# er.test ('data/test')

start ()