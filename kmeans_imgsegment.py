## Neal Trischitta ##
#!/usr/bin/env python2.7
from sys import argv
from PIL import Image
import numpy
import math
import random
import argparse


# Return k centroids RGB triples. 
def get_centroids(im,k):
	cents=[]
	for i in range(k):
		x=random.randrange(im.size[0])
		y=random.randrange(im.size[1])
		cents.append(im.getpixel((x,y)))
	return cents

# Finds the nearest centroid and set the custers
def argmindistance(pixel, centroids):
	distance=[]
	for c in centroids:
		distance.append(math.sqrt((c[0]-pixel[0])**2+(c[1]-pixel[1])**2+(c[2]-pixel[2])**2))
	minindex=numpy.argmin(distance)
	clusters[minindex].append(pixel)
	return minindex

def getAverageRGB(arr):
	r = 0
	g = 0
	b = 0
	counter=len(arr)
	for x in xrange(len(arr)):
		r+=arr[x][0]
		g+=arr[x][1]
		b+=arr[x][2]
	rAvg = r/counter
	gAvg = g/counter
	bAvg = b/counter
	return (rAvg, gAvg, bAvg)

def kmeans(im,k):
	h=im.size[0]
	w=im.size[1]

	centroids=get_centroids(im,k)
	oldcentroids= [[] for i in xrange(k)]
	labels=numpy.zeros((h,w))

	while not(centroids == oldcentroids):
		for x in xrange(h):
			for y in xrange(w):
				labels[x][y]=argmindistance(im.getpixel((x,y)),centroids)

		index=0
		for cluster in clusters:
			if(len(cluster) !=0):
				oldcentroids[index] = centroids[index]
				centroids[index]=getAverageRGB(cluster)
				index+=1

	print("Image has Coverged")
	centers=numpy.array(centroids)
	labels=numpy.transpose(numpy.uint8(labels))
	res=centers[labels]
	Image.fromarray(res.astype('uint8')).show()
#	im.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Kmeans Clustering: Image Segmenation")
	parser.add_argument('-i', '--source_img', help='Source Image', required=True)
	parser.add_argument('-k',  '--bin_size', type=int, help='Bin size', required=True)
	args = parser.parse_args()

	##Open image for processing in RGB colorspace.
	im = Image.open(args.source_img).convert('RGB')
	global clusters
	clusters= [[] for i in range(args.bin_size)]
	kmeans(im,args.bin_size)
