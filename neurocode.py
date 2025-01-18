import numpy as np
import os
import matplotlib.pyplot as plt

class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax and quick group indexing."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def select(self, idx):
        """B.select(idx) returns a new bunch instance with all members indexed by idx"""
        return Bunch({k:self[k][idx] for k in self})
    
    def __len__(self):
        """len(b) checks all members have the same length and returns it. If they don't it raises an error."""
        lengths = [len(self[k]) for k in self]
        if np.max(lengths)!=np.min(lengths):
            raise ValueError('Members of Bunch have different lengths!')
        return lengths[0]
        
    def save(self, filebase):
        """object.save(filebase) saves the object as a collection of files
        for each attribute X, the attribute object.X is saved as the file filebase.X.npy"""
        
        # check lengths before saving
        self.__len__()
        
        # save all files
        for f in self:
            np.save(filebase + '.' + f + '.npy', self[f])
        
def load_object(path):
    r"""
    load_object(path)
    
    Loads the files corresponding to an ONE object (e.g. spikes.times, spikes.clusters) into a Bunch
    
    Parameters
    ----------
    path: string pointing to the object name, without attribute or extension.  
    
    Returns
    -------
    obj: a Bunch containing all loaded files.  
    
    Note
    ----
    For example, to load C:\Users\kenneth\data\spikes.times and C:\Users\kenneth\data\spikes.clusters,
    type load_object('C:\Users\kenneth\data\spikes') and it will return a Bunch with members times and clusters
    
    By convention, all members of an ONE object should have the same length.  A warning will be given if they
    do not"""
    
    head, tail =os.path.split(path)

    ldir = os.listdir(head)
    obj = Bunch()
    for f in ldir:
        if f.startswith(tail + '.') and f.endswith('.npy'): 
            full_filename = os.path.join(head, f)
            if f.count('.')!=2:
                raise ValueError('%s is not a 3-part filename'%full_filename)
            _,a,_ = f.split('.')
            obj[a]=np.load(full_filename, allow_pickle=True)
            
    lengths = [len(obj[k]) for k in obj]
    if min(lengths) != max(lengths):
        warnings.warn('Lengths of attributes in %s are not equal'%full_filename)
        
    return obj

    
def find_close_pairs(x,y, t_before, t_after):
    """
    find_close_pairs(x,y, t_before, t_after)
        Takes two sorted numerical arrays x, y and returns an array all of index pairs (i,j)
        such that -t_before<=x(i)-y(j)<=t_after
    
    Parameters
    ----------
    x, y: 1d numerical arrays, that should be sorted in ascending order
    t_before, t_after: numbers. Must be >=0 for the algo to work 
    
    Returns
    -------
    index_pairs: a Nx2 array of indexes (i,j) such that -t_before<=x(i)-y(j)<=t_after    
    
    Notes
    ----
    It will run more efficiently if y has fewer entries than x.
    
    If you want to search with [t_before,t_after] interval that doesn't include zero, then
    shift x or y instead.
    """
    
    assert np.all(np.diff(x)>=0), "x should be sorted ascending"
    assert np.all(np.diff(y)>=0), "y should be sorted ascending"
    assert t_before>=0, "t_before should be >=0"
    assert t_after>=0, "t_after should be >=0"
    
    # Find i0 for each j largest so that x(i0-1)<y(j)
    i0 = np.searchsorted(x,y)
    

    
    # store a list of (i,j) in pair_list
    pair_list = []
    
    #loop through shifts left from the i0 positions keeping going while -t_before<=x(i)-y(j)
    shift = 0
    # all elements of y for which we will keep searching
    #y_alive = np.ones_like(y, dtype=bool)
    y_alive = (i0<x.shape[0])
    
    while y_alive.any():
        # find all y from those currently alive for which x[i0+shift] is in range
        alive_idx = np.nonzero(y_alive)[0]
        in_range = (-t_before <= (x[i0[alive_idx]+shift] - y[alive_idx]))
        idx_in_range = alive_idx[in_range]
        pair_list.append(np.stack((i0[idx_in_range]+shift,idx_in_range)))

        # remove any out of range from alive
        y_alive[alive_idx[~in_range]] = False
        # remove any that have hit the left edge of x from alive
        y_alive[alive_idx[i0[alive_idx]+shift==0]] = False
        shift -= 1
                         
    # now loop through shifts right from the i0 positions keeping going while x(i)-y(j)<=t_after
    shift = 1 # don't count 0 twice
    # all elements of y for which we will keep searching
    y_alive = (i0<len(x)-shift)
    
    while y_alive.any():
        # find all y from those currently alive for which x[i0+shift] is in range
        alive_idx = np.nonzero(y_alive)[0]
        in_range = (x[i0[alive_idx]+shift] - y[alive_idx] <= t_after)
        idx_in_range = alive_idx[in_range]
        pair_list.append(np.stack((i0[idx_in_range]+shift, idx_in_range)))

        # remove any out of range from alive
        y_alive[alive_idx[~in_range]] = False
        # remove any that have hit the right edge of x from alive
        y_alive[alive_idx[i0[alive_idx]+shift==len(x)-1]] = False
        
        
        shift += 1
                         
    # concatenate the output array and sort
    index_pairs = np.concatenate(pair_list,1)
    index_pairs = index_pairs[:,np.lexsort(index_pairs)]
    
    return index_pairs.T

def plot_raster(spike_times, stim_times, before_time, after_time, sort_by=None, event_types=None, *args, **kwargs):
    pairs = find_close_pairs(spike_times, stim_times, before_time, after_time)
    reltimes = spike_times[pairs[:,0]] - stim_times[pairs[:,1]]
    repeats = pairs[:,1]
    if sort_by is None:
        trial_order=np.arange(repeats.max()+1)
    else:
        trial_order = np.argsort(np.argsort(sort_by)) # get the rank of each repeat
        
    if event_types is None:
        plt.scatter(reltimes, trial_order[repeats], marker='|', *args, **kwargs)
    else:
        event_names, event_ids = np.unique(event_types, return_inverse=True)
        cmap = plt.cm.tab10(np.arange(len(event_names)))
        plt.scatter(reltimes, trial_order[repeats], c=cmap[event_ids[repeats],:], 
                    marker='|', *args, **kwargs)
        
    plt.xlim([-before_time, after_time])
    plt.ylim([trial_order.min(), trial_order.max()])
    plt.gca().invert_yaxis()
    
    plt.plot([0,0], plt.ylim(), 'k')
    
def compute_psth(spike_times, event_times, before_time, after_time, bins=51, spike_clusters=None, event_types=None):
    """return psth: size bins x clusters x event_types
    psth_bin_centers
    event_names
    """
    if np.isscalar(bins):
        psth_bin_edges = np.linspace(-before_time,after_time,bins+1)
    else:
        psth_bin_edges = bins
        
    if event_types is None: event_types = np.zeros_like(event_times)
    if spike_clusters is None: spike_clusters= np.zeros_like(spike_times)

    event_names, event_ids = np.unique(event_types, return_inverse=True)
        
    psth_bin_centers = (psth_bin_edges[1:] + psth_bin_edges[:-1])/2
    pairs = find_close_pairs(spike_times, event_times, before_time, after_time)

    rel_times = spike_times[pairs[:,0]]-event_times[pairs[:,1]]
    
    
    psth = np.histogramdd((rel_times, spike_clusters[pairs[:,0]], event_ids[pairs[:,1]]),
                       bins=[psth_bin_edges, np.arange(spike_clusters.max()+2),np.arange(event_ids.max()+2)]
                      )[0]
                          
    event_count = np.histogram(event_ids,bins=np.arange(event_ids.max()+2))[0]                      
    psth = psth/np.diff(psth_bin_edges)[:,None,None]/event_count[None,None,:]

    return psth, psth_bin_centers, event_names