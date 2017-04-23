# This implements Emotion Recognition using
# "Recognition of Facial Expressions and Measurement of Levels of Interest From Video"
# by M.Yeasin, B.Bullot and R.Sharma.

import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.externals import joblib
import hmmlearn

class EmotionRecognizer:

	def __init__ (self):
		self.neighbors = 10
		self.weights = 'uniform'
		self.ncomponents = 40

		self.temporalSequence = []
		self.blockSizes = []
		self.blockTags = []

		self.knn = None
		self.hmms = None
		
		self.emotions = {'surprise':0, 'happy':1, 'sad':2,
						 'anger':3,	'fear':4, 'disgust':5}

		self.face_cascade = cv2.CascadeClassifier ('data/haarcascade_frontalface_default.xml')


	# From a video this will calculate optical flow and returns
	# a 2d array of optical flows each 1D array in that 2d array will 
	# have x and y components of optical flows side by side

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

		nr, nc = faces.shape
		if nr != 1:
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

			(nr, nc) = faces.shape
			if nr != 1:
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
	# singular vector decomposition 
	def encodeVideo (self, fileName):
		flows = self.calculateFlow (fileName)

		x_std = StandardScaler().fit_transform (flows)

		# v is an array of eigen vectors and s is an array of eigen values
		u, s, v = np.linalg.svd (x_std)

		eig_vecs = v[:,0:40]

		# making transformation using eigen vectors by matrix multiplication
		y = x_std.dot(eig_vecs)

		return y

	# generates temporal sequences for all videos and store it in class variable
	# which will be used to train HMM.
	def genTemporalSequence (self, dataDir):

		# list of flows for each image flow in all videos
		allFlows = []
		# list of tags for each image flow in all videos
		tags = []
		self.blockSizes = []
		self.blockTags = []

		tagname = None

		for fileName in os.listdir (dataDir):
			try:
				tagname = emotion [fileName.split ('_')[1]]
			except 	KeyError as ke:
				print ('no emotion for file')
				continue

			tflows = self.encodeVideo (dataDir + '/' + fileName)

			i = 0
			flows = []
			for flow in tflows:
				if i%5 == 0:
					allFlows.append (flow)
					tags.append (tagname)
				i = i+1

			self.blockSizes.append (tflows.size[0])
			self.blockTags.append (tagname)

		allFlows = np.array (allFlows)
		tags = np.array (tags)

		# train KNN using optical flows and their emotions
		self.knn = neighbors.KNeighborsClassifier(self.nneighbours, weights=self.weights)
		self.knn.fit (allFlows, tags)

		# test "training data" and generate a temporal sequence
		self.temporalSequence = self.knn.predict (test)

		return True

	def train (self, dataDir):
		self.genTemporalSequence (dataDir)
		train = [[],[],[],[],[],[]]

		pcum = 0
		cum = 0
		for emotion, sizes in self.blockTags, self.blockSizes:
			cum = cum + size
			train[emotion].append (self.temporalSequence[pcum:cum-1])
			pcum = cum

		self.genTemporalSequence = []

		self.hmms = [None, None, None, None, None, None]

		for i in range(0,6):
			# hmm.learn (
	
	# saves models into disk
	def dump (self, modelDir):
		try:
			joblib.dump (self.knn, modelDir + '/knn.dump')
			for key, value in self.emotions
				joblib.dump (self.hmm[value], modelDir + '/hmm'+key+'.dump')
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
		except Exception as e:
			print ('Error in loading models')
			return False
		return True

	def testFile (self, fileName, vis):
		pass

	def test (self, fileName):
		if os.path.isfile (fileName):
			for file in os.listdir (fileName):
				self.testFile (fileName, False)
		else:
			self.testFile (fileName, True)

def start ():
	er = EmotionRecognizer ()
	er.train ('data/train')
	er.dump ('data/model')
	# er.test ('data/test')

	print('video blocks:\n', er.blocks)
	print('temporal sequence:\n', er.temporalSequence)

start ()