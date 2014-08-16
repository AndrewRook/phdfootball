import warnings
import numpy as np
import skimage.measure as ski_measure
import skimage.transform as ski_trans
import skimage.io as ski_io
import skimage.exposure as ski_exp
import skimage.util as ski_util
import os
import sys
import bottleneck as bn

from alignlib import compute_frameorder,frame,match_frames_twosided,compute_extrema,adj_mask_or,insert_image

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit('Syntax: [Video Filename] [First Frame #] [Last Frame #] [Frames to Skip (1 for no skip)] [Output Images Prefix] [Print Original Images (Y/N)]')
    
    #Setup video and frame range
    filename = sys.argv[1]
    cap = ski_io.Video(filename)
    firstframe = int(sys.argv[2])
    lastframe = int(sys.argv[3])
    startframe = firstframe + (lastframe-firstframe)//2
    frameskip = int(sys.argv[4])
    outprefix = sys.argv[5]
    printorig = False
    if sys.argv[6].lower() == 'y':
        printorig = True

    #Determine the order which to attach the frames:
    frameorder = compute_frameorder(firstframe,lastframe,startframe,frameskip)
    chronological_order = np.argsort(frameorder)
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
    print "Aligning Frames:"
    for i in range(1,len(frameorder)):
        print "Currently aligning frame {0:d} of {1:d} (frame number {2:d}), master image array shape =".format(i,len(frameorder)-1,frameorder[i]),imagearr.shape
        newframe = frame(cap,frameorder[i])
        #Output the original frames, if necessary:
        if printorig:
            ski_io.imsave(outprefix+'_orig_{0:d}.jpg'.format(frameorder[i]),newframe.rawimage)
        #Compute matches:
        matches = match_frames_twosided(currframe,newframe,dist_ratio=0.6)
        matching_coords_curr = currframe.coords[matches > 0,:]
        matching_coords_new = newframe.coords[matches[matches>0],:]
        #Compute homology:
        model,inliers = ski_measure.ransac((matching_coords_curr,matching_coords_new),ski_trans.ProjectiveTransform,min_samples=25,residual_threshold=1.0,max_trials=2000)
        homologies[i,:,:] = model._matrix
        #Determine where the new frame extrema would be after being transformed:
        xmin,xmax,ymin,ymax = compute_extrema(newframe,model._matrix)
        jointimage = ski_exp.rescale_intensity(ski_util.img_as_float(currframe.rawimage.copy()))
        #Make sure things haven't gone totally awry:
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
        newweight = adj_mask_or((newweight==False),adjustment=5)
        newweight = (newweight == False)
        unwarped_newimage[newweight == False] = 0
        unwarped_newimage = ski_exp.rescale_intensity(ski_util.img_as_float(unwarped_newimage))
        imagearr[:,:,:,i] = unwarped_newimage
        xvals = np.arange(newweight.shape[1])
        yvals = np.arange(newweight.shape[0])
        X,Y = np.meshgrid(xvals,yvals)
        badx = X[newweight == False]
        bady = Y[newweight == False]
        imagearr[bady,badx,:,i] = np.nan
        avg_image = jointimage*weightimage.reshape(weightimage.shape+(1,))+unwarped_newimage*newweight.reshape(newweight.shape+(1,))
        weightimage += newweight
        avg_image /= weightimage.reshape(weightimage.shape+(1,))
        
        #Replace the currframe with the avg_image:
        currframe = frame(cap,-1,image=ski_util.img_as_ubyte(avg_image))
    ski_io.imsave(outprefix+'_avg.jpg',currframe.rawimage)
    medianed_image = np.zeros(imagearr[:,:,:,0].shape)
    medianed_image[:,:,0] = bn.nanmedian(imagearr[:,:,0,:],axis=2)
    medianed_image[:,:,1] = bn.nanmedian(imagearr[:,:,1,:],axis=2)
    medianed_image[:,:,2] = bn.nanmedian(imagearr[:,:,2,:],axis=2)
    ski_io.imsave(outprefix+'_median.jpg',medianed_image)

    #Output the rectified frames:
    for i in range(len(chronological_order)):
        curridx = chronological_order[i]
        print frameorder[curridx],
        currimage = insert_image(medianed_image,imagearr[:,:,:,curridx],adjustmask=1,overlap=20)
        ski_io.imsave(outprefix+'_rectified_{0:d}.jpg'.format(frameorder[curridx]),currimage)

