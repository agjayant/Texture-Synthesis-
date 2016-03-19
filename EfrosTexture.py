#!/usr/bin/python

import cv2
import numpy as np

#Image Loading and initializations
img_sample = cv2.imread("sample_texture.jpg")
img_height = 256
img_width = 256
#empty_pixel = np.zeros((1,1,3), np.uint8)
sample_height = img_sample.shape[0]
sample_width = img_sample.shape[1]
img = np.zeros((img_height,img_width,3), np.uint8)
WindowSize = 3
Sigma = 6.4/WindowSize
flag = -1
boundary = []
FilledPx = np.zeros((img_height,img_width),int)

for i in range(sample_height):
    for j in range(sample_width):
        img[i + 10,j + 20] = img_sample[i,j]        #offset of (10,20)
        FilledPx[i + 10,j + 20] = 1;
        neighbors = GetBoundaryPxls( (i + 10, j + 20), img )
        for px in neighbors:
            x,y = px
            if (x - 10 > sample_height or x - 10 < 0 ):
                boundary.append(px)
            elif ( y - 20 > sample_width or y - 20 < 0 ):
                boundary.append(px)

def GetBoundaryPxls( px, Image ):
    x,y = px
    left = max( 0, x - 1 )
    right = min( Image.shape[0], x + 2 )
    top = max( 0, y - 1 )
    bot = min( Image.shape[1], y + 2 )
    bound = Image[left:right,top:bot]
    bound.remove(px)
    return bound


#Function to return PixelList
def PixelList(img):
    pxlList = []
    height,width  = img.shape
    for i in range(height):
        for j in range(width):
            if FilledPx[i,j] == 0:
                pxlList.append((i,j))
    return pxlList

def GetNeighbourWindow(px, Image, WindowSize):
    x,y = px
    left = max( 0, x - WindowSize/2 )
    right = min( Image.shape[0], x + WindowSize/2 )
    top = max( 0, y - WindowSize/2 )
    bot = min( Image.shape[1], y + WindowSize/2 )
    return Image[ left : right, top : bot ]             #CAUTION : FIX IT...the range
 
def FindMatches(Template, SampleImage):

	Matches=[]
	temp = 0

	# parameters to be customized
	# harcoded for 3
	gauss = 2
	thresh = 1	

	height,width = SampleImage.shape
	for i in range(height):
		for j in range(width):

			temp = 0

			for k in range(3):
				for l in range(3):
 								
					if k == 1 and l==1 :
						temp= temp + gauss*(Template[k, l] - SampleImage[i+k , j+l])**2
					else : 
						temp= temp + (Template[k, l] - SampleImage[i+k , j+l])**2
					
			
			if temp < thresh: 
				Matches.append(SampleImage[i:i+2,j:j+2])

	return Matches	

def RandomPick( MatchList ):
    return random.randrange(0, len(MatchList), 1)

def error( px_match, px, WindowSize, Image ):
    ssd = 0
    x1,y1 = px_match
    x2,y2 = px
    left  = min( WindowSize/2 , x1, x2 )
    right = min( WindowSize/2, Image.shape[0] - max( x1, x2 ) )
    top = min( WindowSize/2, y1, y2 )
    bot = min( WindowSize, Image.shape[1] - max( y1, y2 ) )
    for i in range(left,right):
        for j in range(top,bot):
            temp = Image[ x1 + i, y1 + j] - Image[ x2 + i, y2 + j]
            ssd += (temp[0]^2 + temp[1]^2 + temp[2]^2) * FilledPx[ x2 + i, y2 + j ]
    return ssd

# def GetUnfilledNeighbours( Image, EmptyPixels ):
    # if flag == -1:
        # somthing
   # elif flag == 0:
       # smthng els
    # else: 
        # finl

def GrowImage(SampleImage, Image, WindowSize):
    EmptyPixels = PixelList(Image)
    MaxErrThreshold = 0.3
    while len(EmptyPixels) > 0 :
        progress = 0
        for px in boundary:
            Template = GetNeighbourWindow(px, Image, WindowSize)       
            BestMatches = FindMatches(Template, SampleImage)
            #Finds best matches from sample
            BestMatch = RandomPick(BestMatches)
            if error( BestMatch, px, WindowSize, Image) < MaxErrThreshold:
                px = BestMatch
                progress = 1
                EmptyPixels.remove(px)
                FilledPx[px] = 1
        if progress == 0:
            MaxErrThreshold *= 1.1
    return Image


#Displaying Images
# cv2.imshow('Sample Texture',img_sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Generated Image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
