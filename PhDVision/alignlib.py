import warnings
import numpy as np
import scipy.linalg
import scipy.interpolate
import cv2
import skimage.measure as ski_measure
import skimage.transform as ski_trans
import skimage.io as ski_io
import skimage.exposure as ski_exp
import skimage.util as ski_util
import matplotlib.pyplot as plt
import os
import sys
import bottleneck as bn

import astropy.io.fits as pyfits

class frame:
    def __init__(self,cap,framenumber,panoramafile = None,image = None):
        self.framenumber = framenumber
        if image != None:
            self.rawimage = image
        else:
            if panoramafile != None:
                self.rawimage = ski_io.imread(panoramafile)
            else:
                self.rawimage = cap.get_index_frame(self.framenumber)
        self.grayimage = cv2.cvtColor(self.rawimage,cv2.COLOR_BGR2GRAY)
        self.projmatrix = np.zeros((3,3))
        #Compute coordinates:
        self.coords,self.keypoints,self.descriptors = self.compute_coords()


    def compute_coords(self):
        sift = cv2.SIFT()
        keypoints,descriptors = sift.detectAndCompute(self.grayimage,None)
        coords = np.array([keypoint.pt for keypoint in keypoints])
        return coords,keypoints,descriptors

def match_frames(frame1,frame2,dist_ratio=0.6):
    desc1 = np.array([frame1.descriptors[i,:]/np.linalg.norm(frame1.descriptors[i,:]) for i in range(frame1.descriptors.shape[0])])
    desc2 = np.array([frame2.descriptors[i,:]/np.linalg.norm(frame2.descriptors[i,:]) for i in range(frame2.descriptors.shape[0])])

    matches = np.zeros((desc1.shape[0],1),dtype=np.int)
    desc2_t = desc2.T
    #print desc1.shape
    for i in range(desc1.shape[0]):
        dotprods = 0.9999*np.dot(desc1[i,:],desc2_t)
        arccos_dotprods = np.arccos(dotprods)
        indx = np.argsort(arccos_dotprods)

        if arccos_dotprods[indx[0]] < dist_ratio*arccos_dotprods[indx[1]]:
            matches[i] = int(indx[0])

    return matches
def match_frames_twosided(frame1,frame2,dist_ratio=0.6):
    matches_12 = match_frames(frame1,frame2,dist_ratio)
    matches_21 = match_frames(frame2,frame1,dist_ratio)

    ndx_12 = matches_12.nonzero()[0]

    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12[:,0]

def match_frames_flann(frame1,frame2,dist_ratio=0.8):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(frame1.descriptors,frame2.descriptors,k=2)
    
    goodmatches = np.zeros(len(frame1.descriptors),dtype=np.int)
    for i,(m,n) in enumerate(matches):
        if m.distance < dist_ratio*n.distance:
            goodmatches[m.queryIdx] = m.trainIdx

    return goodmatches

def appendimages(im1,im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    #print im1.shape,im2.shape,rows1,rows2
    if rows1 < rows2:
        #im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
        im1 = np.vstack((im1,np.zeros((rows2-rows1,im1.shape[1],im1.shape[2]))))
    elif rows1 > rows2:
        im2 = np.vstack((im2,np.zeros((rows1-rows2,im2.shape[1],im2.shape[2]))))
        #im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    #print im1.shape,im2.shape,rows1,rows2

    return np.concatenate((im1,im2),axis=1)

def plot_matches(im1,im2,locs1,locs2,matches,show_below=False,ax=None):
    im1_copy = ski_exp.rescale_intensity(ski_util.img_as_float(im1.copy()))
    im2_copy = ski_exp.rescale_intensity(ski_util.img_as_float(im2.copy()))
    im3 = appendimages(im1_copy,im2_copy)
    if show_below:
        im3 = np.vstack((im3,im3))

    if ax == None:
        ax = plt.figure().add_subplot(111)
    ax.imshow(im3)
    cols1 = im1_copy.shape[1]
    # print im3.shape
    # print locs1.shape,locs2.shape,len(matches)
    count = 0
    for i in range(len(matches)):
        ax.plot([locs1[i,0],locs2[i,0]+cols1],[locs1[i,1],locs2[i,1]],'c')
        count += 1
    print "Number of plotted matches=",count,len(matches)
    ax.set_xlim(0,im3.shape[1])
    ax.set_ylim(im3.shape[0],0)
    return ax


def compute_frameorder(first,last,start,skip=1):
    framelist = []
    allframes = np.arange(first,last+skip,skip)
    usedframes = np.zeros(len(allframes),dtype=np.bool)
    startingidx = np.argsort(np.abs(start-allframes))[0]
    startingframe = allframes[startingidx]
    framelist.append(startingframe)
    usedframes[startingidx] = True
    while np.sum(usedframes) < len(usedframes):
        sortedidxs = np.argsort(np.abs(startingframe-allframes))
        nextidx = sortedidxs[usedframes[sortedidxs] == False][0]
        framelist.append(allframes[nextidx])
        usedframes[nextidx] = True
    return np.array(framelist,dtype=np.int)

def compute_extrema(inpframe,homology):
    xvals = np.arange(inpframe.rawimage.shape[1])
    yvals = np.arange(inpframe.rawimage.shape[0])
    X,Y = np.meshgrid(xvals,yvals)
    coords = np.vstack((X.reshape(-1),Y.reshape(-1))).T
    transform = ski_trans.ProjectiveTransform(matrix=homology)
    transformed_coords = transform.inverse(coords)
    return transformed_coords[:,0].min(),transformed_coords[:,0].max(),transformed_coords[:,1].min(),transformed_coords[:,1].max()

def insert_image(panorama,image,adjustmask=1, overlap=5):
    #Get the region where only 1 image should be present:
    goodbool = (np.isnan(image[:,:,0]) == False)
    goodbool = adj_mask(goodbool,adjustment=adjustmask)
    #Create an interpolation between the individual frame and the panorama:
    interp_vals = interp_mask(goodbool,offset=overlap)
    interp_vals = interp_vals.reshape(interp_vals.shape+(1,))
    #Insert the image into the panorama:
    imagenans = np.isnan(image)
    zeroed_image = image.copy()
    zeroed_image[imagenans] = 0
    combimage = panorama*interp_vals + zeroed_image*(1.-interp_vals)
    return combimage

def interp_mask(boolmask,offset=5):
    adj_bool = adj_mask(boolmask,adjustment=offset)
    interp_bool = (adj_bool == False) & (boolmask == True)
    big_interp_bool = adj_mask_or(interp_bool,adjustment=1)
    edge_interp_bool = (big_interp_bool == True) & (interp_bool == False)

    interp_vals = np.ones(boolmask.shape,dtype=np.float)
    interp_vals[adj_bool] = 0.
    xvals = np.arange(interp_vals.shape[1])
    yvals = np.arange(interp_vals.shape[0])
    X,Y = np.meshgrid(xvals,yvals)
    goodx = X[edge_interp_bool == True]
    goody = Y[edge_interp_bool == True]
    goodpoints = np.vstack((goodx,goody)).T
    interpx = X[interp_bool == True]
    interpy = Y[interp_bool == True]
    interppoints = np.vstack((interpx,interpy)).T
    goodvals = interp_vals[edge_interp_bool == True].astype(np.float)
    print len(goodvals)
    if len(goodvals) > 0:
        interpdata = scipy.interpolate.griddata(goodpoints,goodvals,interppoints,method='linear',fill_value=0.)
    else:
        interpdata = interpx*0 + 1
    interp_vals[interpy,interpx] = interpdata

    return interp_vals
    # os.system('rm -f test*.fits')
    # hdu = pyfits.PrimaryHDU((interp_bool).astype(np.int))
    # hdu.writeto('test_interp.fits')
    

def adj_mask(inpmask,adjustment=1):
    boolmask = inpmask.copy()
    if adjustment > 0:
        boolmask[:-adjustment,:] = boolmask[:-adjustment,:] & boolmask[adjustment:,:]
        boolmask[adjustment:,:] = boolmask[:-adjustment,:] & boolmask[adjustment:,:]
        boolmask[:,:-adjustment] = boolmask[:,:-adjustment] & boolmask[:,adjustment:]
        boolmask[:,adjustment:] = boolmask[:,:-adjustment] & boolmask[:,adjustment:]
    return boolmask

def adj_mask_or(inpmask,adjustment=1):
    boolmask = inpmask.copy()
    if adjustment > 0:
        boolmask[:-adjustment,:] = boolmask[:-adjustment,:] | boolmask[adjustment:,:]
        boolmask[adjustment:,:] = boolmask[:-adjustment,:] | boolmask[adjustment:,:]
        boolmask[:,:-adjustment] = boolmask[:,:-adjustment] | boolmask[:,adjustment:]
        boolmask[:,adjustment:] = boolmask[:,:-adjustment] | boolmask[:,adjustment:]
    return boolmask

def homology_chisquared(im1,im2,matrix_1d):
    matrix = matrix_1d.reshape(3,3)
    transform = ski_trans.ProjectiveTransform(matrix=matrix)
    unwarped_image = ski_trans.warp(im1,transform,output_shape=im2.shape,cval=-10)
    unwarped_image[unwarped_image < 0] = np.nan
    #print np.sum(unwarped_image < 0),unwarped_image.shape[0]*unwarped_image.shape[1]*unwarped_image.shape[2]
    resid_image = unwarped_image - im2
    resids = resid_image[np.isnan(resid_image) == False]
    #print len(resids),resid_image.shape[0]*resid_image.shape[1]*resid_image.shape[2]
    return resid_image[np.isnan(resid_image) == False]
    
if __name__ == "__main__":
    #Load video
    #Select first frame, last frame, and the starting frame
    #Take in starting frame, compute coords using SIFT
    #Take in an adjacent frame, compute coords using SIFT
    #Match the two frames, compute the homology
    #Determine where the corners of all the frames would be if the new frame was warped onto the old
    #Translate the first frame to accomodate the new frame, then translate the new frame onto the new space
    #Combine the adjacent frame onto the first frame, ideally using a median
    #Use the combined frame as the first frame above, adding new frames on 1 by 1.

    #TODO: EXPERIMENT WITH ADJUSTING RANSAC ERROR LIMIT, MAKING IT LARGER MIGHT ACTUALLY HELP
    #ADJUST MATCHING DISTANCE LIMIT
    #ADD LM FITTER FOR ADJUSTING FITS
    
    #Setup video and frame range
    filename = '/Users/Rook/nflstats/nflvids/nyj/2013/2013100700.mp4'
    cap = ski_io.Video(filename)
    firstframe = 8750#2690
    lastframe = 8890#2750
    startframe = 8820#2720
    frameskip = 1

    #Determine the order which to attach the frames:
    frameorder = compute_frameorder(firstframe,lastframe,startframe,frameskip)
    chronological_order = np.argsort(frameorder)
    print frameorder
    #Initialize the array for the homologies:
    homologies = np.zeros((len(frameorder),3,3))
    #initialize the first one to be a zero transform:
    homologies[0,0,0] = 1.
    homologies[0,1,1] = 1.
    homologies[0,2,2] = 1.
    #Load up the first frame:
    currframe = frame(cap,frameorder[0])
    weightimage = np.ones(currframe.grayimage.shape)
    imagearr = np.zeros((currframe.rawimage.shape[0],currframe.rawimage.shape[1],currframe.rawimage.shape[2],len(frameorder)))/0.
    imagearr[:,:,:,0] = ski_exp.rescale_intensity(ski_util.img_as_float(currframe.rawimage))
    #Go through the frames:
    for i in range(1,len(frameorder)):
        print i,len(frameorder)-1,frameorder[i],imagearr.shape
        newframe = frame(cap,frameorder[i])
        #Compute matches:
        matches = match_frames_twosided(currframe,newframe,dist_ratio=0.6)
        #matches = match_frames_flann(currframe,newframe)
        matching_coords_curr = currframe.coords[matches > 0,:]
        matching_coords_new = newframe.coords[matches[matches>0],:]
        #Compute homology:
        model,inliers = ski_measure.ransac((matching_coords_curr,matching_coords_new),ski_trans.ProjectiveTransform,min_samples=25,residual_threshold=1.0,max_trials=2000)
        #print currframe.rawimage.max(),currframe.rawimage.min()
        test = homology_chisquared(newframe.rawimage,ski_exp.rescale_intensity(ski_util.img_as_float(currframe.rawimage)),model._matrix.reshape(-1))
        #print np.sum(test),test.min(),test.max()
        #print len(inliers),np.sum(inliers)
        homologies[i,:,:] = model._matrix
        #Determine where the new frame extrema would be after being transformed:
        xmin,xmax,ymin,ymax = compute_extrema(newframe,model._matrix)
        jointimage = ski_exp.rescale_intensity(ski_util.img_as_float(currframe.rawimage.copy()))
        #ax = plot_matches(newframe.rawimage,currframe.rawimage,matching_coords_new[inliers,:],matching_coords_new[inliers,:],matches[inliers],show_below=False,ax=None)
        #ax.figure.savefig('test_matches_{0:d}.jpg'.format(newframe.framenumber))
        #print xmin,ymin,xmax,ymax,imagearr.shape,np.sum(inliers)
        if xmax > currframe.rawimage.shape[1]*1.5 or ymax > currframe.rawimage.shape[0]*1.5 or xmin < currframe.rawimage.shape[1]*-0.5 or ymin < currframe.rawimage.shape[0]*-0.5:
            ski_io.imsave('test_avg_aborted.jpg',currframe.rawimage)
            print "    ",xmin,xmax,ymin,ymax,currframe.rawimage.shape
            sys.exit("Image matching has likely failed, aborting to avoid running out of memory")
        #Buffer the current frame and adjust homologies:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if xmax > currframe.rawimage.shape[1]:
                buf = int(np.ceil(xmax-currframe.rawimage.shape[1]))
                jointimage = np.hstack((jointimage,np.zeros((jointimage.shape[0],buf,3))/0.))
                imagearr = np.hstack((imagearr,np.zeros((imagearr.shape[0],buf,imagearr.shape[2],imagearr.shape[3]))/0.))
                weightimage = np.hstack((weightimage,np.zeros((weightimage.shape[0],buf))))
            if ymax > currframe.rawimage.shape[0]:
                buf = int(np.ceil(ymax-currframe.rawimage.shape[0]))
                jointimage = np.vstack((jointimage,np.zeros((buf,jointimage.shape[1],3))/0.))
                imagearr = np.vstack((imagearr,np.zeros((buf,imagearr.shape[1],imagearr.shape[2],imagearr.shape[3]))/0.))
                weightimage = np.vstack((weightimage,np.zeros((buf,weightimage.shape[1]))))
            if ymin < 0:
                buf = int(np.ceil(np.abs(ymin)))
                homologies[:,1,2] -= buf
                jointimage = np.vstack((np.zeros((buf,jointimage.shape[1],3))/0.,jointimage))
                imagearr = np.vstack((np.zeros((buf,imagearr.shape[1],imagearr.shape[2],imagearr.shape[3]))/0,imagearr))
                weightimage = np.vstack((np.zeros((buf,weightimage.shape[1])),weightimage))
            if xmin < 0:
                buf = int(np.ceil(np.abs(xmin)))
                homologies[:,0,2] -= buf
                jointimage = np.hstack((np.zeros((jointimage.shape[0],buf,3))/0.,jointimage))
                imagearr = np.hstack((np.zeros((imagearr.shape[0],buf,imagearr.shape[2],imagearr.shape[3]))/0,imagearr))
                weightimage = np.hstack((np.zeros((weightimage.shape[0],buf)),weightimage))
        #Warp the new image onto the joint image:
        transform = ski_trans.ProjectiveTransform(matrix=homologies[i,:,:])
        unwarped_newimage = ski_trans.warp(newframe.rawimage,transform,output_shape=jointimage.shape,cval=-10)
        newweight = (unwarped_newimage[:,:,0] >= 0)
        #print newweight.dtype,newweight.min(),newweight.max()
        newweight = adj_mask_or((newweight==False),adjustment=5)
        newweight = (newweight == False)
        #print newweight.dtype,newweight.min(),newweight.max()
        unwarped_newimage[newweight == False] = 0
        unwarped_newimage = ski_exp.rescale_intensity(ski_util.img_as_float(unwarped_newimage))
        imagearr[:,:,:,i] = unwarped_newimage
        xvals = np.arange(newweight.shape[1])
        yvals = np.arange(newweight.shape[0])
        X,Y = np.meshgrid(xvals,yvals)
        badx = X[newweight == False]
        bady = Y[newweight == False]
        imagearr[bady,badx,:,i] = np.nan
        #med_image = np.zeros(imagearr[:,:,:,-1].shape)
        #med_image[:,:,0] = bn.nanmedian(imagearr[:,:,0,:i+1],axis=2)
        #med_image[:,:,1] = bn.nanmedian(imagearr[:,:,1,:i+1],axis=2)
        #med_image[:,:,2] = bn.nanmedian(imagearr[:,:,2,:i+1],axis=2)
        avg_image = jointimage*weightimage.reshape(weightimage.shape+(1,))+unwarped_newimage*newweight.reshape(newweight.shape+(1,))
        weightimage += newweight
        avg_image /= weightimage.reshape(weightimage.shape+(1,))
        
        #Replace the currframe with the avg_image:
        currframe = frame(cap,-1,image=ski_util.img_as_ubyte(avg_image))
        #currframe = frame(cap,-1,image=ski_util.img_as_ubyte(med_image))
    ski_io.imsave('test_avg.jpg',currframe.rawimage)
    medianed_image = np.zeros(imagearr[:,:,:,0].shape)
    medianed_image[:,:,0] = bn.nanmedian(imagearr[:,:,0,:],axis=2)
    medianed_image[:,:,1] = bn.nanmedian(imagearr[:,:,1,:],axis=2)
    medianed_image[:,:,2] = bn.nanmedian(imagearr[:,:,2,:],axis=2)
    ski_io.imsave('test_median.jpg',medianed_image)

    for i in range(len(chronological_order)):
        curridx = chronological_order[i]
        print frameorder[curridx],
        currimage = insert_image(medianed_image,imagearr[:,:,:,curridx],adjustmask=1,overlap=20)
        ski_io.imsave('test_out_{0:d}.jpg'.format(frameorder[curridx]),currimage)

