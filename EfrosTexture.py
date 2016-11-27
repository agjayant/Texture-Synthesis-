#!/usr/bin/python
import cv2
import numpy as np
import random
import scipy.ndimage.filters as fi
import sys

#Image Loading and initializations
img_sample = cv2.imread(sys.argv[1])

sample_height = img_sample.shape[0]
sample_width = img_sample.shape[1]

img_height = sample_height + 10
img_width = sample_width + 10

img = np.zeros((img_height,img_width,3), np.uint8)
WindowSize = int(sys.argv[2])
Sigma = WindowSize/6.4 				###################### Parameter

FilledPx = np.zeros((img_height,img_width),int)

for i in range(sample_height):
    for j in range(sample_width):
        img[i + 5,j + 5] = img_sample[i,j]        #offset of (5,5)
        FilledPx[i + 5,j + 5] = 1; 


def PixelList(img):
    pxlList = []
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            if FilledPx[i,j] == 0:
                pxlList.append((i,j))
    return pxlList

def GetNeighbourWindow(px, Image, WindowSize):
    y,x = px
    left = max( 0, x - WindowSize/2 )
    right = min( Image.shape[1], x + WindowSize/2 )
    top = max( 0, y - WindowSize/2 )
    bot = min( Image.shape[0], y + WindowSize/2 )
    return Image[ top : bot,left:right ]

def FindMatches(Template, SampleImage):

	#Valid Mask
	h_template = Template.shape[0]
	w_template = Template.shape[1]

	ValidMask= np.zeros((h_template,w_template),int)

	for k in range(h_template):
		for l in range(w_template):
			if sum(Template[k,l])> 0:
				ValidMask [k,l] = 1

	inp = np.zeros((h_template,w_template))
	inp[h_template//2,w_template//2] =1          ########################## Parameter
	GaussMask = fi.gaussian_filter(inp,Sigma)

	# Total Weight
	TotWeight = sum(sum(GaussMask*ValidMask))
	
	height= SampleImage.shape[0]
	width = SampleImage.shape[1]

	SSD = np.zeros((height,width))
	minSSD = 100000      #infinity

	for i in range(h_template/2,height-h_template/2):
		top = max(i-h_template/2, 0 )
		
		for j in range(w_template/2, width-w_template/2):
			
			left = max(j- w_template/2, 0)

			for k in range(h_template):
				for l in range(w_template):
					a = Template[k,l]
					b = SampleImage[top+k,left+l]										
					dist = (int(a[0])-int(b[0]))**2+ (int(a[1])-int(b[1]))**2+ (int(a[2])-int(b[2]))**2 
					SSD[i,j]= SSD[i,j] + (dist*ValidMask[k,l]*GaussMask[k,l])

			if SSD[i,j] > 0 :
				SSD[i,j] = SSD[i,j]/TotWeight

			if SSD[i,j] < minSSD :
				minSSD = SSD[i,j]

	ErrThreshold = 0.1       ################################################ Parameter

	Matches = []

	for i in range(height):
		for j in range(width):
			if SSD[i,j] <= minSSD*(1+ErrThreshold):
				Matches.append((i,j))
	
	return Matches

def RandomPick( MatchList ):
    return MatchList[random.randrange(0, len(MatchList), 1)]

def error(px_match, px_tofill, SampleImage , Image ):
	
	sam = np.zeros((WindowSize,WindowSize,3),np.uint8)
	fill = np.zeros((WindowSize,WindowSize,3),np.uint8)
	
	x,y =px_match

	top= max(0,y-WindowSize/2)
	bot= min(SampleImage.shape[0],y+WindowSize/2 )
	left= max(0,x-WindowSize/2)
	right=min(SampleImage.shape[1],x+WindowSize/2)

	for i in range(top,bot):
		for j in range(left,right):

			k = i - (y-WindowSize/2)
			l = j - (x-WindowSize/2)
			sam[k,l] = SampleImage[i,j]

	x,y =px_tofill

	top= max(0,y-WindowSize/2)
	bot= min(Image.shape[0],y+WindowSize/2 )
	left= max(0,x-WindowSize/2)
	right=min(Image.shape[1],x+WindowSize/2)

	for i in range(top,bot):
		for j in range(left,right):

			k = i - (y-WindowSize/2)
			l = j - (x-WindowSize/2)
			fill[k,l] = Image[i,j]


	inp = np.zeros((WindowSize,WindowSize))
	inp[WindowSize//2,WindowSize//2] =0.1          ###################################  Parameter
	GaussMask = fi.gaussian_filter(inp,Sigma)

	ssd = 0
	for i in range(WindowSize):
		for j in range(WindowSize):
			a = sam[i,j]
			b = fill[i,j]
			temp = 0
			if sum(a) + sum(b) > 0:
				temp = (int(a[0])-int(b[0]))**2+ (int(a[1])-int(b[1]))**2+ (int(a[2])-int(b[2]))**2 
				temp = (temp*GaussMask[i,j])

			ssd = ssd + temp
	return ssd

def GetBoundary(Image):
	boundary =[]
	height= Image.shape[0]
	width = Image.shape[1]

	for i in range(height):
		for j in range(width):
			if FilledPx[i,j] == 0:

				left = max(0,j-1)
				top = max(0,i-1)
				right = min(j+1,width-1)
				bot = min(i+1,height-1)

				count = 0

				for k in range(left,right+1):
					for l in range(top,bot+1):
						if FilledPx[l,k] ==1:
							count += 1

				if count>0:
					boundary.append((count,(i,j)))
	
	boundary.sort(reverse=True)
	return boundary			 
	

def GrowImage(SampleImage, Image, WindowSize):
    EmptyPixels = PixelList(Image)
    MaxErrThreshold = 1.5                    ############################### Parameter
    while len(EmptyPixels) > 0 :
        progress = 0
	boundary = GetBoundary(Image)
        for pix in boundary:
	    px=pix[1]
            Template = GetNeighbourWindow(px, Image, WindowSize)       
            BestMatches = FindMatches(Template, SampleImage)
            BestMatch = RandomPick(BestMatches)
            if error( BestMatch, px, SampleImage, Image) < MaxErrThreshold:
	          Image[px] = SampleImage[BestMatch]
        	  progress = 1
	          EmptyPixels.remove(px)
                  FilledPx[px] = 1
                  print BestMatch

	    print len(EmptyPixels)
	    
        if progress == 0:
            MaxErrThreshold *= 1.4
	    print MaxErrThreshold        #############################Parameter


    return Image


img  = GrowImage(img_sample,img,WindowSize)
#Displaying Images
#cv2.imshow('Sample Texture',img_sample)
#cv2.imshow('Generated Image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("result.png",img)
