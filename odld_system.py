from __future__ import print_function, division
import numpy, time
from west.propagators import WESTPropagator
from west.systems import WESTSystem
from westpa.binning import RectilinearBinMapper, VoronoiBinMapper
from scipy.spatial.distance import cdist

PI = numpy.pi
from numpy import sin, cos, exp
from numpy.random import normal as random_normal
import numpy as np

pcoord_len = 21
pcoord_dtype = np.float32    

def dfunc(p, centers):
    #print("Dfunc called")
    #print(p, centers)
    #print(p.shape, centers.shape)
    ds = cdist(np.array([p]),centers)
    return np.array(ds[0], dtype=p.dtype)

class ODLDPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(ODLDPropagator,self).__init__(rc)
        
        self.coord_len = pcoord_len
        self.coord_dtype = pcoord_dtype
        self.coord_ndim = 2
        
        self.initial_pcoord = np.array([2.0,2.0], dtype=self.coord_dtype)
        
        self.sigma = 0.1
        
        self.well_1 = [2,2,2,0.6,0.6]
        self.well_2 = [3,8,8,0.6,0.6]
        self.well_3 = [1,7.1,2,0.5,20]

        self.x0 = np.ones(self.coord_ndim)
        
        # Implement a reflecting boundary at this x value
        # (or None, for no reflection)
        self.reflect_at_1 = 10.0
        self.reflect_at_2 = 0.0
        #self.reflect_at = None

    def get_pcoord(self, state):
        '''Get the progress coordinate of the given basis or initial state.'''
        state.pcoord = self.initial_pcoord.copy()
                
    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    def grad_x(self, A, sig_x, sig_y, x0, y0, x, y):
        var_x = 2*sig_x*sig_x
        var_y = 2*sig_y*sig_y
        pre = (2*A * (x-x0))/var_x
        ex = np.exp(((-1/var_x)*(x-x0)**2)+((-1/var_y)*(y-y0)**2))
        return pre*ex
    
    def grad_y(self, A, sig_x, sig_y, x0, y0, x, y):
        var_x = 2*sig_x*sig_x
        var_y = 2*sig_y*sig_y
        pre = (2*A * (y-y0))/var_y
        ex = np.exp(((-1/var_x)*(x-x0)**2)+((-1/var_y)*(y-y0)**2))
        return pre*ex

    def propagate(self, segments):
        start_time = time.time()
        
        well_1, well_2, well_3 = self.well_1, self.well_2, self.well_3
        sig = self.sigma
        gradfactor = 2*sig*sig
        cos45 = np.cos(np.pi/4)
        sin45 = np.sin(np.pi/4)
        
        n_segs = len(segments)
    
        coords = np.empty((n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype)
        
        for iseg, segment in enumerate(segments):
            coords[iseg,:] = segment.pcoord[:]
            
        coord_len = self.coord_len
        reflect_at_1 = self.reflect_at_1
        reflect_at_2 = self.reflect_at_2
        all_displacements = np.zeros((n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype)
        for istep in xrange(1,coord_len):
            x = coords[:,istep-1,:]
           
            all_displacements[:,istep,:] = displacements = random_normal(scale=sig, size=(n_segs,self.coord_ndim))
            # We have 3 wells
            mod_x = (cos45*x[:,0]+sin45*x[:,1])
            mod_y = (cos45*x[:,0]-sin45*x[:,1])
            grad_x = self.grad_x(well_1[0], well_1[3], well_1[4],  well_1[1], well_1[2], x[:,0], x[:,1]) + \
                     self.grad_x(well_2[0], well_2[3], well_2[4],  well_2[1], well_2[2], x[:,0], x[:,1]) + \
                     self.grad_x(well_3[0], well_3[3], well_3[4],  well_3[1], well_3[2], mod_x, mod_y)
            grad_y = self.grad_y(well_1[0], well_1[3], well_1[4],  well_1[1], well_1[2], x[:,0], x[:,1]) + \
                     self.grad_y(well_2[0], well_2[3], well_2[4],  well_2[1], well_2[2], x[:,0], x[:,1]) + \
                     self.grad_y(well_3[0], well_3[3], well_3[4],  well_3[1], well_3[2], mod_x, mod_y)
            grad = np.array([grad_x,grad_y]).T
            
            #print("x : ", x)
            #print("gf: ", gradfactor)
            #print("g: ", grad)
            #print("disp: ", displacements)
            newx = x - gradfactor*grad + displacements
            #print("newxs: ", newx)
            if reflect_at_1 is not None:
                # Anything that has moved beyond reflect_at must move back that much
                
                # boolean array of what to reflect
                to_reflect_1 = newx > reflect_at_1
                to_reflect_2 = newx < reflect_at_2
                
                # how far the things to reflect are beyond our boundary
                reflect_by_1 = newx[to_reflect_1] - reflect_at_1
                reflect_by_2 = newx[to_reflect_2] - reflect_at_2
                
                # subtract twice how far they exceed the boundary by
                # puts them the same distance from the boundary, on the other side
                newx[to_reflect_1] -= 2*reflect_by_1
                newx[to_reflect_2] -= 2*reflect_by_2
            coords[:,istep,:] = newx
        cputime_end = time.time()
            
        for iseg, segment in enumerate(segments):

            segment.pcoord[...] = coords[iseg,:]
            segment.data['displacement'] = all_displacements[iseg]
            segment.cputime = cputime_end - start_time
            end_time = time.time()
            segment.walltime = end_time - start_time
            segment.status = segment.SEG_STATUS_COMPLETE
    
        return segments

class ODLDSystem(WESTSystem):
    def initialize(self):
        self.pcoord_ndim = 2
        self.pcoord_dtype = pcoord_dtype
        self.pcoord_len = pcoord_len
        
        nbins = 1
        self.nbins = nbins

        centers = np.zeros((self.nbins,self.pcoord_ndim),dtype=self.pcoord_dtype)
        centers[:,:] = 1

        self.bin_mapper = VoronoiBinMapper(dfunc, centers)
        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int)
        self.bin_target_counts[...] = 10
