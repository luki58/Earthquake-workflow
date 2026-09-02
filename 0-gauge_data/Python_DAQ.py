"""
Python Functions to use a DAQ
"""

## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate
# DAQ
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# random
import threading
import pickle
import time
import os

## Constants
E=3.48e9
nu=0.3

## Functions
def open_data_bin_double(location):
    """
    Obsolete
    Used to open data from the oscilloscopes
    See open_data_bin
    """
    try :
        from DAQ_Python.importAgilentBin import readfile
    except :
        from importAgilentBin import readfile
    time,data_1=readfile(location,0)
    _,data_2=readfile(location,1)
    return(time,data_1,data_2)

def open_data_bin(location):
    """
    Opens binary data from the agilent oscilloscopes
    Return a tuple containing the timescale and an array of the data of each
    """
    try :
        from DAQ_Python.importAgilentBin import readfile
    except :
        from importAgilentBin import readfile
    data_n=None
    data_tot=[]
    time=None
    bool_stop=False
    n=0
    while not bool_stop:
        time_temp,data_n=readfile(location,n)
        n+=1
        bool_stop= data_n is None
        if not bool_stop:
            data_tot.append(data_n)
            time=time_temp
    return(time,np.array(data_tot))



def avg_bin(data,n):
    """
    Binning and averaging over the bins of size n
    The new data is of length len(data)//n
    """
    l=len(data)
    new_l=l//n
    data_avg=data[:new_l*n].reshape(new_l,n)
    data_avg=np.mean(data_avg,axis=1)
    return(data_avg)


def median_bin(data,n):
    """
    Binning and applying a median over the bins of size n
    The new data is of length len(data)//n
    """
    l=len(data)
    new_l=l//n
    data_avg=data[:new_l*n].reshape(new_l,n)
    data_avg=np.median(data_avg,axis=1)
    return(data_avg)


def V_to_strain(data,amp=2000,G=1.79,i_0=0.0017,R=350):
    """
    Applies the conversion from Voltage to Strain
    amp is the amplification factor
    G is the gauge factor
    i_0 is the fixed current
    R is the gauge resistance
    """
    return(data/(amp*R*G*i_0))

def strain_to_V(data,amp=495,G=1.79,i_0=0.0017,R=350):
    """
    Applies the conversion from strain to voltage
    amp is the amplification factor
    G is the gauge factor
    i_0 is the fixed current
    R is the gauge resistance
    """
    return(data*(amp*R*G*i_0))



def rosette_to_tensor(ch_1,ch_2,ch_3):
    """
    converts a 45 degres rosette signal into a full tensor.
    input : the three channels of the rosette
    output : $\epsilon_{xx},\epsilon_{yy},\epsilon_{xy}
    https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/strain_gage_rosette.cfm
    """
    eps_xx=ch_1+ch_3-ch_2
    eps_yy=ch_2
    eps_xy=(ch_1-ch_3)/2
    return(eps_xx,eps_yy,eps_xy)


def cart2pol(x, y):
    """
    Converts polar coordinates into cartesian
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(r, theta)

def pol2cart(r, theta):
    """
    Converts cartesian coordinates into polar
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return(x, y)

def theta_x(theta,alpha_x):
    """
    Intermediary function
    """
    return(np.arctan(alpha_x*np.tan(theta))%np.pi)

def gamma(theta,v,c):
    """
    Used to compute the fracture energy
    """
    return(np.sqrt(1-(v*np.sin(theta)/c)**2))



def import_struc_from_matlab(location):
    """
    This function imports a matlab .mat file containing a single structure
    It gives a dictionnary as an output.
    """
    lib = scipy.io.loadmat(location)
    table=lib[list(lib.keys())[-1]]
    keys=eval(str(table.dtype))
    keys=np.array(keys)[:,0]
    dict={}
    for i in range(len(keys)):
        value=table[0][0][i]
        try :
            if value.shape==(1,):
                value=value[0]
            elif value.shape==(1,1):
                value=value[0][0]
            elif value.shape[0]==1:
                value=value[0]
            elif value.shape[-1]==1:
                value=value.squeeze(axis=-1)
        except:
            pass
        dict[keys[i]]=value
    return(dict)


def concatarrays(arr1,arr2,permutation=None):
    """concatenates two arrays on the shortest one, on the first dimension"""
    s1=arr1.shape
    s2=arr2.shape
    if s2[1]<s1[1]:
        return(np.concatenate([arr1[:,:s2[1]],arr2]))
    elif s1[1]==s2[1]:
        return(np.concatenate([arr1,arr2]))
    else:
        return(np.concatenate([arr1,arr2[:,:s1[1]]]))
    return(None)


def jump_plot(signal,jumps):
    """
    Just a ploting device
    """
    plt.plot(signal)
    for i in jumps:
        plt.axvline(i,linestyle="--")
    plt.show()

def jump_detect(signal,typical_wait_time=25,sensit=5):
    """
    Uses a simple differentiation to find abrupt jumps in data
    typical_wait_time is the typical time between two events, it eliminates events that are too close one to each other
    """
    diff=np.diff(signal)
    diff=np.abs(diff)
    m=np.mean(diff)
    diff=diff-m
    s=1#np.min([np.std(diff[i:i+typical_wait_time]) for i in range(len(diff)-typical_wait_time)])
    index=np.where(diff>sensit*s)[0]
    index=list(index)
    index.append(index[-1]+2*typical_wait_time)
    index=np.array(index)
    diffindex=np.diff(index)
    indexreal=index[np.where(diffindex>typical_wait_time)[0]]
    return(indexreal)


def time_to_index(times_list,time):
    """
    Finds the position of each elements of a list into a x_data
    """
    index_list=[]
    times_list=sorted(times_list)
    j=0
    for t in times_list:
        while time[j]<t:
            j+=1
        index_list.append(j)
    return(index_list)

def retime_data(time,time_out,data_out):
    """
    Used to reevaluate data on a different time axis.
    time : time on which you want your data
    time_out : time at which the data was originaly taken
    data_out : the data you want to reevaluate

    Example :
                I have {t_i}_{i\in I}
                I have {y_i}_{i\in I} = f({t_i}_{i\in I})
                I want {y_j}_{j\in J} = f({t_j}_{j\in J}) with I != J
                `retime_data([t_j for j in J], [t_i for i in I], [])`
    """
    if time[0]<time_out[0]:
        time_out=np.insert(time_out,0,time[0])
        data_out[0]=0
        data_out=np.insert(data_out,0,0)

    if time[-1]>time_out[-1]:
        time_out=np.append(time_out,time[-1])
        data_out[-1]=0
        data_out=np.append(data_out,0)

    new_data=np.empty_like(time)
    f = interpolate.interp1d(time_out, data_out)
    new_data = f(time)
    return(new_data)


def load_params(loc_params):
    with open(loc_params, 'r') as f:
        code = f.read()
    return(code)



def voltage_to_strains(ch1,ch2,ch3,amp=2000):
    """
    Converts three strain gages channels from bare voltage to real strain
    CH_2 (eps_yy) IS INVERTED TO ENSURE THAT LOADING IS ASSOCIATED WITH INCREASINF EPS.
    """
    # apply different G coefficient to the sides and the center, according
    # to the gages documentation.
    side=lambda x: V_to_strain(x,amp=amp,G=1.79,i_0=0.0017,R=350)
    center=lambda x: -V_to_strain(x,amp=amp,G=1.86,i_0=0.0017,R=350)
    ch1=side(ch1)
    ch2=center(ch2)
    ch3=side(ch3)
    return(ch1,ch2,ch3)

def voltage_to_force(ch):
    """
    Converts Doerler force sensor from voltage to force in kg
    """
    # correct sign
    for i in range(len(ch)):
        ch[i]=ch[i]*np.median(np.sign(ch[i]))
    #return(500/3*ch)
    #2025-29-01-EB-I might have changed the calibration of the force sensor last december to recover the full range of measurements. Then it affects the calibration. Now 10V for 1000 kg.
    return((1000/5.6)*ch)


def list_files_with_pattern(directory, pattern):
    matching_files = []
    import fnmatch
    for file_name in os.listdir(directory):
        if fnmatch.fnmatch(file_name, f'*{pattern}*'):
            matching_files.append(file_name)

    return(sorted(matching_files))




def eps_to_sigma(eps_xx,eps_yy,eps_xy,E=E,nu=nu):
    """
    Converts epsilon to sigma with a plane stress hypothesis
    """
    a=E/(1-nu**2)
    b=nu*a
    c=E/(1+nu)
    sigma_xx=a*eps_xx+b*eps_yy
    sigma_yy=b*eps_xx+a*eps_yy
    sigma_xy=c*eps_xy
    return(sigma_xx,sigma_yy,sigma_xy)





def trigger_to_binary(trig,start=10000,limit=3):
    sm_trig = trig
    sm_trig=sm_trig>limit
    sm_trig[4:-1]+=sm_trig[0:-5]
    sm_trig=sm_trig>0.5
    stop = len(sm_trig)
    j=start
    events=[]
    while j < stop:
        if sm_trig[j] :
            k=0
            while j+k<stop and sm_trig[j+k]:
                k+=1
            if k>25:
                events.append(j)
            j=j+k
        else :
            j+=1
    return(events)





def trig_to_TTL(trigger):
    sm_trig=trigger>3
    sm_trig[4:-1]+=sm_trig[0:-5]
    sm_trig=sm_trig>0.5
    stop = len(sm_trig)
    j=10
    timings=[]
    while j < stop:
        if sm_trig[j] :
            k=0
            while j+k<stop and sm_trig[j+k]:
                k+=1
            if k>25:
                timings.append(j)
            j=j+k
        else :
            j+=1
    sm_trig=np.zeros_like(trigger)
    sm_trig[timings]+=1
    return(sm_trig)




def gauge_number_to_channel(n):
    """
    Converts the gauge number of the JPBox into the actual channel number in the
    PeterBox data.
    """
    try :
        _=len(n)
        n=np.array(n)
        n[ n>45 ]+=2
        n[ np.logical_and( 30<n , n<=45 ) ]+=1
        n[ n<=15 ]-=1
        return(n)
    except :
        if n<=15:
            return(n-1)
        elif n<=30:
            return(n)
        elif n<=45:
            return(n+1)
        else:
            return(n+2)

def PeterBox_Channels_To_JPBox_Categories(data):
    """
    Takes demuxed data from the PeterBox and converts it to gages data and front channels data, separately
    """
    gauges_number = np.arrange(1,61,1)
    gauges_channel= gauges_number_to_channel(gauges_number)
    front_channel = [15,31,47,63]
    gauges_data   = data[gauges_channel]
    front_data    = data[front_channel]
    return(gauges_data,front_data)




def rolling_average(a, n=10) :
    """
    A simple denoising trick using rolling average.
    A more elaborate harmonic filtering could be useful.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return(ret[n - 1:] / n)


def smooth(a, n=10) :
    """
    A simple denoising trick using rolling average.
    A more elaborate harmonic filtering could be useful.
    """
    ret = np.cumsum(a, axis=-1, dtype=float)
    ret[...,n:] = ret[...,n:] - ret[...,:-n]
    return( ret[...,n - 1:] / n )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


















