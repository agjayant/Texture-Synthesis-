#!/usr/bin/python
import cv2
import numpy as np
import random
import scipy.ndimage.filters as fi

#Image Loading and initializations
img_sample = cv2.imread("final.png")
#img_height = 256
#img_width = 256
#empty_pixel = np.zeros((1,1,3), np.uint8)

sample_height = img_sample.shape[0]
sample_width = img_sample.shape[1]

img_height = sample_height + 10
img_width = sample_width + 10

img = np.zeros((img_height,img_width,3), np.uint8)
WindowSize = 21
Sigma = WindowSize/6.4
flag = -1
#boundary = []
FilledPx = np.zeros((img_height,img_width),int)
#Matches=[]

'''
def GetBoundaryPxls( px, Image ):
    x,y = px
    left = max( 0, x - 1 )
    right = min( Image.shape[0], x + 2 )
    top = max( 0, y - 1 )
    bot = min( Image.shape[1], y + 2 )
    #bound = Image[left:right,top:bot]
    #bound.remove(px)
    bound =[]
    for i in range(left,right+1):
	    for j in range(bot,top+1):
		    bound.append((i,j))			     
    np.delete(bound,px)
    return bound


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

'''

for i in range(sample_height):
    for j in range(sample_width):
        img[i + 5,j + 5] = img_sample[i,j]        #offset of (5,5)
        FilledPx[i + 5,j + 5] = 1; 

'''
def GetBoundary(px):
    x,y = px
    boundary.append((x-1,y-1))
    i = x
    while FilledPx[i,y] == 1:
	    boundary.append((i,y-1))
	    i= i+1
    right  = i
    boundary.append(end,y-1)
    
    j = y
    while FilledPx[x,j] ==1 :
	    boundary.append((x-1,j))
	    j = j+1
    bot = j

    boundary.append(x-1,bot)

    for i in range(x,end+1):
	    boundary.append((i,bot))

    for j in range(y,bot):
	    boundary.append(end,j)
	
for i in range(sample_height):
    for j in range(sample_width):
        img[i + 10,j + 20] = img_sample[i,j]        #offset of (10,20)
        FilledPx[i + 10,j + 20] = 1;
                
GetBoundary((10,20))
'''

#Function to return PixelList
def PixelList(img):
    pxlList = []
    #height,width  = img.shape
    height = img.shape[0]
    width = img.shape[1]
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

	#Valid Mask
	h_template = Template.shape[0]
	w_template = Template.shape[1]

	ValidMask= np.zeros((h_template,w_template),int)

	for k in range(h_template):
		for l in range(w_template):
			if sum(Template[k,l])> 0:
				ValidMask [k,l] = 1

	inp = np.zeros((h_template,w_template))
	inp[h_template//2,w_template//2] =10 
	GaussMask = fi.gaussian_filter(inp,Sigma)

	# Total Weight
	TotWeight = sum(sum(GaussMask*ValidMask))
	
	height= SampleImage.shape[0]
	width = SampleImage.shape[1]

	SSD = np.zeros((height,width))
#	print "hello"
	for i in range(height-h_template):
		for j in range(width-w_template):

			for k in range(h_template):
				for l in range(w_template):
										
					dist = (Template[k,l]- SampleImage[i+k,j+l])**2
					SSD[i,j]= SSD[i,j] + sum(dist*ValidMask[k,l]*GaussMask[k,l])
	#				print SSD[i,j]

			if SSD[i,j] > 0 :
				SSD[i,j] = SSD[i,j]/TotWeight

	ErrThreshold = 0.1       #########################  To be played with
	minSSD= np.amin(SSD)

	Matches = []

	for i in range(height):
		for j in range(width):
			if SSD[i,j] <= minSSD*(1+ErrThreshold):
				Matches.append(SampleImage[i,j])
	
	return Matches

def RandomPick( MatchList ):
    return MatchList[random.randrange(0, len(MatchList), 1)]

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

def GetBoundaryNaive(Image):
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

				flag = 0
				for k in range(left,right+1):
					for l in range(top,bot+1):
						if FilledPx[l,k] ==1:
							flag =1
							break
					if flag ==1 :
						break

				if flag ==1:
					boundary.append((i,j))
	
	return boundary
				 
	

def GrowImage(SampleImage, Image, WindowSize):
    EmptyPixels = PixelList(Image)
    MaxErrThreshold = 0.3
    while len(EmptyPixels) > 0 :
        #progress = 0
	boundary = GetBoundaryNaive(Image)
        for px in boundary:
            Template = GetNeighbourWindow(px, Image, WindowSize)       
            BestMatches = FindMatches(Template, SampleImage)
            #Finds best matches from sample
	    #print len(BestMatches)
            BestMatch = RandomPick(BestMatches)
         #  if error( BestMatch, px, WindowSize, Image) < MaxErrThreshold:
            Image[px] = BestMatch
         #       progress = 1
            EmptyPixels.remove(px)
            FilledPx[px] = 1
	    print BestMatch
	    print len(boundary)
	    print len(EmptyPixels)
        #if progress == 0:
        #    MaxErrThreshold *= 1.1

    return Image


img  = GrowImage(img_sample,img,WindowSize)
#Displaying Images
#cv2.imshow('Sample Texture',img_sample)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('Generated Image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("result.png",img)
