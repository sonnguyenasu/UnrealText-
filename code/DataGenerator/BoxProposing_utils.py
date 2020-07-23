import numpy as np
import cv2
from ransac import fit_plane_ransac
from text_utils import RenderFont
import scipy.spatial.distance as ssd
import itertools
min_char_height = 8
min_asp_ratio = 0.4
text_renderer = RenderFont('data')
def depth2xyz(depth):
        """
        Convert a HxW depth image (float, in meters)
        to XYZ (HxWx3).

        y is along the height.
        x is along the width.
        """
        H,W = depth.shape
        xx,yy = np.meshgrid(np.arange(W),np.arange(H))
        X = (xx-W/2) * depth / 520
        Y = (yy-H/2) * depth / 520
        return np.dstack([X,Y,depth.copy()])
def isplanar(xyz,sample_neighbors,dist_thresh,frac_inliers,z_proj, returnCoeffs= False):
        """
        Checks if at-least FRAC_INLIERS fraction of points of XYZ (nx3)
        points lie on a plane. The plane is fit using RANSAC.

        XYZ : (nx3) array of 3D point coordinates
        SAMPLE_NEIGHBORS : 5xN_RANSAC_TRIALS neighbourhood array
                           of indices into the XYZ array. i.e. the values in this
                           matrix range from 0 to number of points in XYZ
        DIST_THRESH (default = 10cm): a point pt is an inlier iff dist(plane-pt)<dist_thresh
        FRAC_INLIERS : fraction of total-points which should be inliers to
                       to declare that points are planar.
        Z_PROJ : changes the surface normal, so that its projection on z axis is ATLEAST z_proj.

        Returns:
            None, if the data is not planar, else a 4-tuple of plane coeffs.
        """
        if(xyz.shape[0] == 0):                 
            return False
        #frac_inliers = num_inliers/xyz.shape[0]
        dv = -np.percentile(xyz,50,axis=0) # align the normal to face towards camera
        max_iter = sample_neighbors.shape[-1]
        plane_info =  fit_plane_ransac(xyz,neighbors=sample_neighbors,
                                z_pos=dv,dist_inlier=dist_thresh,
                                min_inlier_frac=frac_inliers,nsample=20,
                                max_iter=max_iter) 
        if plane_info != None:
            coeff, inliers = plane_info
            coeff = ensure_proj_z(coeff, z_proj)
            print('xyz inliers:',xyz[inliers].shape)
            print('xyz total:',xyz.shape)
            if returnCoeffs:
                return coeff, inliers
            return True #coeff, inliers
        else:
            return False#None
def ensure_proj_z(plane_coeffs, min_z_proj):
        a,b,c,d = plane_coeffs
        if np.abs(c) < min_z_proj:
            s = ((1 - min_z_proj**2) / (a**2 + b**2))**0.5
            coeffs = np.array([s*a, s*b, np.sign(c)*min_z_proj, d])
            assert np.abs(np.linalg.norm(coeffs[:3])-1) < 1e-3
            return coeffs
        return plane_coeffs
def sample_grid_neighbours(mask,nsample,step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2*step >= min(mask.shape[:2]):
            return #None

        y_m,x_m = np.where(mask)
        mask_idx = np.zeros_like(mask,'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i],x_m[i]] = i

        xp,xn = np.zeros_like(mask), np.zeros_like(mask)
        yp,yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:,:-2*step] = mask[:,2*step:]
        xn[:,2*step:] = mask[:,:-2*step]
        yp[:-2*step,:] = mask[2*step:,:]
        yn[2*step:,:] = mask[:-2*step,:]
        valid = mask&xp&xn&yp&yn

        ys,xs = np.where(valid)
        N = len(ys)
        if N==0: #no valid pixels in mask:
            return #None
        nsample = min(nsample,N)
        idx = np.random.choice(N,nsample,replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs,ys = xs[idx],ys[idx]
        s = step
        X = np.transpose(np.c_[xs,xs+s,xs+s,xs-s,xs-s][:,:,None],(1,2,0))
        Y = np.transpose(np.c_[ys,ys+s,ys-s,ys+s,ys-s][:,:,None],(1,2,0))
        sample_idx = np.concatenate([Y,X],axis=1)
        mask_nn_idx = np.zeros((5,sample_idx.shape[-1]),'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:,i] = mask_idx[sample_idx[:,:,i][:,0],sample_idx[:,:,i][:,1]]
        return mask_nn_idx

def warpHomography(src_mat,H,dst_size):
        dst_mat = cv2.warpPerspective(src_mat, H, dst_size,
                                      flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        return dst_mat

def homographyBB(bbs, H, offset=None):
        """
        Apply homography transform to bounding-boxes.
        BBS: 2 x 4 x n matrix  (2 coordinates, 4 points, n bbs).
        Returns the transformed 2x4xn bb-array.

        offset : a 2-tuple (dx,dy), added to points before transfomation.
        """
        eps = 1e-16
        # check the shape of the BB array:
        t,f,n = bbs.shape
        assert (t==2) and (f==4)

        # append 1 for homogenous coordinates:
        bbs_h = np.reshape(np.r_[bbs, np.ones((1,4,n))], (3,4*n), order='F')
        if offset != None:
            bbs_h[:2,:] += np.array(offset)[:,None]

        # perpective:
        bbs_h = H.dot(bbs_h)
        bbs_h /= (bbs_h[2,:]+eps)

        bbs_h = np.reshape(bbs_h, (3,4,n), order='F')
        return bbs_h[:2,:,:]

def bb_filter(bb0,bb,text):
        """
        Ensure that bounding-boxes are not too distorted
        after perspective distortion.

        bb0 : 2x4xn martrix of BB coordinates before perspective
        bb  : 2x4xn matrix of BB after perspective
        text: string of text -- for excluding symbols/punctuations.
        """
        h0 = np.linalg.norm(bb0[:,3,:] - bb0[:,0,:], axis=0)
        w0 = np.linalg.norm(bb0[:,1,:] - bb0[:,0,:], axis=0)
        hw0 = np.c_[h0,w0]

        h = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        w = np.linalg.norm(bb[:,1,:] - bb[:,0,:], axis=0)
        hw = np.c_[h,w]

        # remove newlines and spaces:
        text = ''.join(text.split())
        assert len(text)==bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        hw0 = hw0[alnum,:]
        hw = hw[alnum,:]

        min_h0, min_h = np.min(hw0[:,0]), np.min(hw[:,0])
        asp0, asp = hw0[:,0]/hw0[:,1], hw[:,0]/hw[:,1]
        asp0, asp = np.median(asp0), np.median(asp)

        asp_ratio = asp/asp0
        is_good = ( min_h > min_char_height
                    and asp_ratio > min_asp_ratio
                    and asp_ratio < 1.0/min_asp_ratio)
        return is_good


def get_min_h(bb, text):
        # find min-height:
        h = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        # remove newlines and spaces:
        text = ''.join(text.split())
        assert len(text)==bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        h = h[alnum]
        return np.min(h)


def feather(text_mask, min_h):
        # determine the gaussian-blur std:
        if min_h <= 15 :
            bsz = 0.25
            ksz=1
        elif 15 < min_h < 30:
            bsz = max(0.30, 0.5 + 0.1*np.random.randn())
            ksz = 3
        else:
            bsz = max(0.5, 1.5 + 0.5*np.random.randn())
            ksz = 5
        return cv2.GaussianBlur(text_mask,(ksz,ksz),bsz)

def place_text(rgb,collision_mask,H,Hinv):
        font = text_renderer.font_state.sample()
        font = text_renderer.font_state.init_font(font)

        render_res = text_renderer.render_sample(font,collision_mask)
        if render_res is None: # rendering not successful
            return #None
        else:
            text_mask,loc,bb,text = render_res

        # update the collision mask with text:
        collision_mask += (255 * (text_mask>0)).astype('uint8')

        # warp the object mask back onto the image:
        text_mask_orig = text_mask.copy()
        bb_orig = bb.copy()
        text_mask = warpHomography(text_mask,H,rgb.shape[:2][::-1])
        bb = homographyBB(bb,Hinv)
        
        if not bb_filter(bb_orig,bb,text):
            #warn("bad charBB statistics")
            return #None
        
        # get the minimum height of the character-BB:
        min_h = get_min_h(bb,text)

        #feathering:
        text_mask = feather(text_mask, min_h)
        global debugger
        
        #im_final = colorizer.color(rgb,[text_mask],np.array([min_h]))
        

        return rgb, text, bb, collision_mask



def get_text_placement_mask(xyz,mask,plane,pad=2,viz=False):
        """
        Returns a binary mask in which text can be placed.
        Also returns a homography from original image
        to this rectified mask.

        XYZ  : (HxWx3) image xyz coordinates
        MASK : (HxW) : non-zero pixels mark the object mask
        REGION : DICT output of TextRegions.get_regions
        PAD : number of pixels to pad the placement-mask by
        """
        contour,hier = cv2.findContours(mask.copy().astype('uint8'),
                                        mode=cv2.RETR_CCOMP,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contour = [np.squeeze(c).astype('float') for c in contour]
        #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
        H,W = mask.shape[:2]

        # bring the contour 3d points to fronto-parallel config:
        pts,pts_fp = [],[]
        center = np.array([W,H])/2
        n_front = np.array([0.0,0.0,-1.0])
        for i in range(len(contour)):
            cnt_ij = contour[i]
            xyz = plane2xyz(center, cnt_ij, plane)
            R = rot3d(plane[:3],n_front)
            xyz = xyz.dot(R.T)
            pts_fp.append(xyz[:,:2])
            pts.append(cnt_ij)

        # unrotate in 2D plane:
        rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
        box = np.array(cv2.boxPoints(rect))
        R2d = unrotate2d(box.copy())
        box = np.vstack([box,box[0,:]]) #close the box for visualization

        mu = np.median(pts_fp[0],axis=0)
        pts_tmp = (pts_fp[0]-mu[None,:]).dot(R2d.T) + mu[None,:]
        boxR = (box-mu[None,:]).dot(R2d.T) + mu[None,:]
        
        # rescale the unrotated 2d points to approximately
        # the same scale as the target region:
        s = rescale_frontoparallel(pts_tmp,boxR,pts[0])
        boxR *= s
        for i in range(len(pts_fp)):
            pts_fp[i] = s*((pts_fp[i]-mu[None,:]).dot(R2d.T) + mu[None,:])

        # paint the unrotated contour points:
        minxy = -np.min(boxR,axis=0) + pad//2
        ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:,0]).T))
        COL = np.max(ssd.pdist(np.atleast_2d(boxR[:,1]).T))

        place_mask = 255*np.ones((int(np.ceil(COL))+pad, int(np.ceil(ROW))+pad), 'uint8')

        pts_fp_i32 = [(pts_fp[i]+minxy[None,:]).astype('int32') for i in range(len(pts_fp))]
        cv2.drawContours(place_mask,pts_fp_i32,-1,0,
                         thickness=cv2.FILLED,
                         lineType=8,hierarchy=hier)
        

        # calculate the homography
        H,_ = cv2.findHomography(pts[0].astype('float32').copy(),
                                 pts_fp_i32[0].astype('float32').copy(),
                                 method=0)

        Hinv,_ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                    pts[0].astype('float32').copy(),
                                    method=0)
        if viz:
            plt.subplot(1,2,1)
            plt.imshow(mask,cmap='binary')
            plt.subplot(1,2,2)
            
            plt.imshow(~place_mask,cmap='binary')
            for i in range(len(pts_fp_i32)):
                plt.scatter(pts_fp_i32[i][:,0],pts_fp_i32[i][:,1],
                            edgecolors='none',facecolor='g',alpha=0.5)
            plt.show()

        return place_mask,H,Hinv
def get_num_text_regions(nregions):
        #return nregions
        nmax = min(7, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0,1.0)
        return int(np.ceil(nmax * rnd))
def filter_for_placement(xyz,seg,coeffs,labels):
        filt = np.zeros(len(labels)).astype('bool')
        masks,Hs,Hinvs = [],[], []
        for idx,l in enumerate(labels):
            #print(seg==l)
            res = get_text_placement_mask(xyz,seg==l,coeffs[idx],pad=2)
            if res is not None:
                mask,H,Hinv = res
                masks.append(mask)
                Hs.append(H)
                Hinvs.append(Hinv)
                filt[idx] = True
        

        return masks, Hs, Hinvs
def plane2xyz(center, ij, plane):
        """
        converts image pixel indices to xyz on the PLANE.

        center : 2-tuple
        ij : nx2 int array
        plane : 4-tuple

        return nx3 array.
        """
        ij = np.atleast_2d(ij)
        n = ij.shape[0]
        ij = ij.astype('float')
        xy_ray = (ij-center[None,:]) / 520
        z = -plane[2]/(xy_ray.dot(plane[:2])+plane[3])
        xyz = np.c_[xy_ray, np.ones(n)] * z[:,None]
        return xyz


def rot3d(v1,v2):
        """
        Rodrigues formula : find R_3x3 rotation matrix such that v2 = R*v1.
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
        """
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1,v2)
        s = np.linalg.norm(v3)
        c = v1.dot(v2)
        Vx = ssc(v3)
        return np.eye(3)+s*Vx+(1-c)*Vx.dot(Vx)

def unrotate2d(pts):
        """
        PTS : nx3 array
        finds principal axes of pts and gives a rotation matrix (2d)
        to realign the axes of max variance to x,y.
        """
        mu = np.median(pts,axis=0)
        pts -= mu[None,:]
        l,R = np.linalg.eig(pts.T.dot(pts))
        R = R / np.linalg.norm(R,axis=0)[None,:]

        # make R compatible with x-y axes:
        if abs(R[0,0]) < abs(R[0,1]): #compare dot-products with [1,0].T
            R = np.fliplr(R)
        if not np.allclose(np.linalg.det(R),1):
            if R[0,0]<0:
                R[:,0] *= -1
            elif R[1,1]<0:
                R[:,1] *= -1
            else:
                print ("Rotation matrix not understood")
                return
        if R[0,0]<0 and R[1,1]<0:
            R *= -1
        assert np.allclose(np.linalg.det(R),1)

        # at this point "R" is a basis for the original (rotated) points.
        # we need to return the inverse to "unrotate" the points:
        return R.T #return the inverse


def ssc(v):
        """
        Returns the skew-symmetric cross-product matrix corresponding to v.
        """
        v /= np.linalg.norm(v)
        return np.array([[    0, -v[2],  v[1]],
                         [ v[2],     0, -v[0]],
                         [-v[1],  v[0],     0]])

def rescale_frontoparallel(p_fp,box_fp,p_im):
        """
        The fronto-parallel image region is rescaled to bring it in 
        the same approx. size as the target region size.

        p_fp : nx2 coordinates of countour points in the fronto-parallel plane
        box  : 4x2 coordinates of bounding box of p_fp
        p_im : nx2 coordinates of countour in the image

        NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

        Returns the scale 's' to scale the fronto-parallel points by.
        """
        l1 = np.linalg.norm(box_fp[1,:]-box_fp[0,:])
        l2 = np.linalg.norm(box_fp[1,:]-box_fp[2,:])

        n0 = np.argmin(np.linalg.norm(p_fp-box_fp[0,:][None,:],axis=1))
        n1 = np.argmin(np.linalg.norm(p_fp-box_fp[1,:][None,:],axis=1))
        n2 = np.argmin(np.linalg.norm(p_fp-box_fp[2,:][None,:],axis=1))

        lt1 = np.linalg.norm(p_im[n1,:]-p_im[n0,:])
        lt2 = np.linalg.norm(p_im[n1,:]-p_im[n2,:])

        s =  max(lt1/l1,lt2/l2)
        if not np.isfinite(s):
            s = 1.0
        return s

def char2wordBB(charBB, text):
        """
        Converts character bounding-boxes to word-level
        bounding-boxes.

        charBB : 2x4xn matrix of BB coordinates
        text   : the text string

        output : 2x4xm matrix of BB coordinates,
                 where, m == number of words.
        """
        wrds = text.split()
        bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
        wordBB = np.zeros((2,4,len(wrds)), 'float32')
        
        for i in range(len(wrds)):
            cc = charBB[:,:,bb_idx[i]:bb_idx[i+1]]

            # fit a rotated-rectangle:
            # change shape from 2x4xn_i -> (4*n_i)x2
            cc = np.squeeze(np.concatenate(np.dsplit(cc,cc.shape[-1]),axis=1)).T.astype('float32')
            rect = cv2.minAreaRect(cc.copy())
            box = np.array(cv2.boxPoints(rect))

            # find the permutation of box-coordinates which
            # are "aligned" appropriately with the character-bb.
            # (exhaustive search over all possible assignments):
            cc_tblr = np.c_[cc[0,:],
                            cc[-3,:],
                            cc[-2,:],
                            cc[3,:]].T
            perm4 = np.array(list(itertools.permutations(np.arange(4))))
            dists = []
            for pidx in range(perm4.shape[0]):
                d = np.sum(np.linalg.norm(box[perm4[pidx],:]-cc_tblr,axis=1))
                dists.append(d)
            wordBB[:,:,i] = box[perm4[np.argmin(dists)],:].T

        return wordBB
