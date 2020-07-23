import numpy as np
import random
from BoxProposing_utils import *
import scipy.spatial.distance as ssd
class BoxProposalModule(object):
    def __init__(self, *args, **kwargs):
        self.MinimapRatio = kwargs.get('MinimapRatio', 2)  # we down-sample the normal boundary map to save computation cost
        self.min_box_size = kwargs.get('min_box_size', 64) // self.MinimapRatio
        self.min_aspect_ratio = kwargs.get('min_aspect_ratio', 0.3)
        self.max_aspect_ratio = kwargs.get('max_aspect_ratio', 10.)
        self.anchor_x_step = kwargs.get('anchor_x_step', 12) // self.MinimapRatio
        self.anchor_y_step = kwargs.get('anchor_y_step', 12) // self.MinimapRatio
        self.SafeGap = kwargs.get('SafeGap', 2) // self.MinimapRatio  # shrink the proposals so that they wont get too close to the boundaries
        self.overlap_threshold = kwargs.get('overlap_threshold', 1)  # threshold for overlap with boundaries in normal maps

    def BoxFitting(self, normal_boundary_map, box_x, box_y):
        """
        Given the center point and normal boundary map
        :param normal_boundary_map: ndarray, HxW, 0/1
        :param box_x: int
        :param box_y: int 
        :return: UL_x, UL_y, BR_x, BR_y
        """
        h, w = normal_boundary_map.shape

        x_left_min = 0
        x_left_max = box_x - self.min_box_size // 2
        x_right_min = box_x + self.min_box_size // 2
        x_right_max = w

        y_down_min = 0
        y_down_max = box_y - self.min_box_size // 4
        y_up_min = box_y + self.min_box_size // 4
        y_up_max = h

        # if already crossing boundary or outside
        if (x_left_max < 0) or \
           (x_right_min >= w) or \
           (y_down_max < 0) or \
           (y_up_min >= h) or \
           (np.sum(normal_boundary_map[y_down_max:y_up_min, x_left_max:x_right_min]) > self.overlap_threshold):
            return -1 

        enlargeable_flag = True
        enlargement_count = 0
        direction = [1, 2, 3, 4]  # right, left, down, up

        while enlargeable_flag:
            random.shuffle(direction)
            enlargeable_flag = False
            for d in direction:
                if d == 4:  # up
                    if y_up_max-y_up_min < 2: continue
                    mid = (y_up_min+y_up_max) // 2
                    if np.sum(normal_boundary_map[y_down_max:mid,
                                         x_left_max:x_right_min]) > self.overlap_threshold:
                        y_up_max = mid
                    else:
                        y_up_min = mid
                    enlargeable_flag = True
                    break
                elif d == 3:  # down
                    if y_down_max-y_down_min < 2: continue
                    mid = (y_down_max+y_down_min) // 2
                    if np.sum(normal_boundary_map[mid:y_up_min,
                                         x_left_max:x_right_min]) > self.overlap_threshold:
                        y_down_min = mid
                    else:
                        y_down_max = mid
                    enlargeable_flag = True
                    break
                elif d == 2:  # left
                    if x_left_max-x_left_min < 2: continue
                    mid = (x_left_max+x_left_min) // 2
                    if np.sum(normal_boundary_map[y_down_max:y_up_min,
                                         mid:x_right_min]) > self.overlap_threshold:
                        x_left_min = mid
                    else:
                        x_left_max = mid
                    enlargeable_flag = True
                    break
                elif d == 1:  # right
                    if x_right_max-x_right_min < 2: continue
                    mid = (x_right_max+x_right_min) // 2
                    if np.sum(normal_boundary_map[y_down_max:y_up_min,
                                         x_left_max:mid]) > self.overlap_threshold:
                        x_right_max = mid
                    else:
                        x_right_min = mid
                    enlargeable_flag = True
                    break
            if enlargeable_flag:
                enlargement_count += 1
            if enlargement_count >= 15:
                if random.random() < 0.2: break

        return x_left_max+self.SafeGap, y_down_max+self.SafeGap, x_right_min-self.SafeGap, y_up_min-self.SafeGap

    def BoxProposing(self, normal_boundary_map, proposalnumber):
        """
        normal_boundary_map: ndarray, HxW 0/1, binary maps
        proposalnumber: how many proposals to return
        """
        H, W = normal_boundary_map.shape
        Proposals = []
        hs = [h for h in range(random.randint(0, self.anchor_y_step // self.MinimapRatio), H, self.anchor_y_step // self.MinimapRatio)]
        random.shuffle(hs)
        ws = [w for w in range(random.randint(0, self.anchor_x_step // self.MinimapRatio), W, self.anchor_x_step // self.MinimapRatio)]
        random.shuffle(ws)
        # random visits
        for h in hs[:]:
            for w in ws[:]:
                result = self.BoxFitting(normal_boundary_map, w, h)
                # filter out failed proposals
                if result == -1:
                    continue
                UL_x, UL_y, BR_x, BR_y = result
                normal_boundary_map[UL_y:BR_y, UL_x: BR_x] = 1
                center_x = (UL_x + BR_x) / 2
                center_y = (UL_y + BR_y) / 2
                result = [center_x - self.min_box_size / 2, 
                          center_y - self.min_box_size / 2, 
                          center_x + self.min_box_size / 2, 
                          center_y + self.min_box_size / 2]
                Proposals.append(np.array(result, dtype=np.int)*self.MinimapRatio)
                #if len(Proposals) >= proposalnumber * 4:
                    #random.shuffle(Proposals)
                    #return Proposals[:proposalnumber]
        random.shuffle(Proposals)
        return Proposals#[:proposalnumber]

    def BoxRefining(self, depth_map, proposalnumber, Proposals):
        idxes = len(Proposals)
        
        labels = np.zeros_like(Proposals, dtype=bool)
        if(len(labels.shape) == 2):
            labels = labels[:,0]
        xyz = depth2xyz(depth_map)
        for idx in range(idxes):
            UL_x, UL_y, BR_x, BR_y = Proposals[idx]            
            
            masks = np.zeros((depth_map.shape[0],depth_map.shape[1]), dtype='uint8')
            masks[UL_y:BR_y,UL_x:BR_x] += 1
            pt = xyz[masks==1]
            
            if pt.shape[0] == 0:
                continue
            pt_sample = sample_grid_neighbours(masks,100)
            labels[idx] = isplanar(pt,pt_sample,0.01,0.80,0.10)
        print(labels)        
        Proposals = np.array(Proposals)[labels]
                
        return Proposals[:proposalnumber]
    def ValidBox(self, depth_map, mask):
        
        
        xyz = depth2xyz(depth_map)
        pt = xyz[mask==1]
            
        if pt.shape[0] == 0:
            return False
        pt_sample = sample_grid_neighbours(mask,100)
        if pt_sample is None:
            return False
        return isplanar(pt,pt_sample,0.01,0.95,0.10)
        #print(labels)        
        #Proposals = np.array(Proposals)[labels]
                
        #return Proposals[:proposalnumber]        
    def SynthTextBox(self,light_map_, depth_map_,seg,Proposals,proposalnumber):
        itext = []
        ibb = []  
        res = []  
        idict = {'img':[], 'charBB':None, 'wordBB':None, 'txt':None}               
        min_char_height = 15
        min_asp_ratio = 0.4
        text_renderer = RenderFont('data')
        for i in range(len(Proposals)):
            try:
                UL_x, UL_y, BR_x, BR_y = Proposals[i]
                depth_map = depth_map_#[UL_y:BR_y,UL_x:BR_x]
                light_map = light_map_#[UL_y:BR_y,UL_x:BR_x]
                xyz = depth2xyz(depth_map)
                seg = np.ones(depth_map.shape)
                masks = seg == 1
                pt_sample = sample_grid_neighbours(masks,1000)
                try:
                    coeffs, inliers = isplanar(xyz[masks],pt_sample,0.10,999,0.1, returnCoeffs= True)
                except: continue           
                coeffs = np.array([coeffs])
                inliers = np.array([inliers])
                labels = np.array([1])
                try:
                    place_masks, Hs, Hinvs = filter_for_placement(xyz,seg,coeffs,labels)
                except:
                    continue
                n_regions = len(place_masks)
                if n_regions < 1: return []
                m = get_num_text_regions(n_regions)
                reg_idx = np.arange(min(2*m,n_regions))
                np.random.shuffle(reg_idx)
                reg_idx = reg_idx[:m]

                placed = False
                img = light_map.copy()
                    

                # process regions: 
                num_txt_regions = len(reg_idx)
                NUM_REP = 5 # re-use each region three times:
                reg_range = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
                for idx in reg_range:
                    ireg = reg_idx[idx]
                    txt_render_res = place_text(img,place_masks[ireg],
                                                                     Hs[ireg],
                                                                     Hinvs[ireg])
                            

                    if txt_render_res is not None:
                        placed = True
                        img,text,bb,collision_mask = txt_render_res
                        # update the region collision mask:
                        place_masks[ireg] = collision_mask
                        # store the result:
                        itext.append(text)
                        print("Placed")
                        bw = char2wordBB(bb.copy(), ' '.join(itext))
                        #import matplotlib.pyplot as plt
                        #plt.imshow(light_map_)
                        #plt.scatter(bw[0,:,:]+UL_x,bw[1,:,:]+UL_y)
                        #plt.show()
                        print(bw)
                        res.append(bw.copy())
                        break
                        
                
            except: continue
        try:        
            res = np.concatenate(res, axis=2)
        except:
            pass
        random.shuffle(res)
        return res[:proposalnumber]



