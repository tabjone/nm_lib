#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

--------------------------------------------------------------------------------------------------

NOTE TO NEW USERS:

--------------------------------------------------------------------------------------------------

"""

# import builtin modules
import os

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt 


def deriv_dnw(xx, hh, **kwargs):
    """
    returns the downwind 2nd order derivative of hh respect to xx. 
    Requires: 
        numpy.roll
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx. 
    Output: 
        returns the downwind 2nd order derivative of hh respect to xx. Last 
        grid point is ill calculated. 
    """


def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function 
    Input: 
        hh  :: function that depends on xx. 
        hh2 :: function that depends on xx but with twice number of grid points than hh. 
        hh4 :: function that depends on xx but with twice number of grid points than hh2.
    Output: 
        returns the order of convergence.  
    """
   

def deriv_4tho(xx, hh, **kwargs): 
    """
    returns the 4th order derivative of hh respect to xx. 
    Requires: 
        numpy.roll
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx. 
    Output: 
        returns the centered 4th order derivative of hh respect to xx. Last 2 and first two 
        grid points are ill calculated. 
    """


def step_adv_burgers(xx, hh, a, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    """
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 
    Requires: 
        cfl_adv_burger function which computes np.min(dx/a)
    Input:    
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: lambda function that allows to select the type of spatial derivative
    Output: 
        dt :: time interval
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """    


def cfl_adv_burger(a,x): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the advective term in the Burger's eq. 
    Requires: 
        numpy gradient, abs, and min
    Input: 
        a :: either constant, or array which multiply the right hand side of the Burger's eq.
        x :: spatial axis. 
    Ouput: 
        min(dx/|a|)
    """


def evolv_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    """
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires: 
         step_adv_burgers
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def deriv_upw(xx, hh, **kwargs):
    """
    returns the upwind 2nd order derivative of hh respect to xx. 
    Requires: 
        numpy.roll
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx. 
    Output: 
        returns the upwind 2nd order derivative of hh respect to xx. First 
        grid point is ill calculated. 
    """
    

def deriv_cent(xx, hh, **kwargs):
    """
    returns the centered 2nd derivative of hh respect to xx. 
    Requires: 
        numpy.roll
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx. 
    Output: 
        returns the centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """


def evolv_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    """
    Advance nt time-steps in time the burger eq for a being u.
    Requires: 
         step_uadv_burgers
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def evolv_Lax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    """
    Advance nt time-steps in time the burger eq for a being u using the Lax method.
    Requires: 
         step_uadv_burgers
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def evolv_Lax_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    """
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires: 
         step_adv_burgers
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def step_uadv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    """
    Right hand side of Burger's eq. where a is u, i.e hh.  
    Requires: 
        cfl_adv_burger function which computes np.min(dx/a)
    Input:    
        xx :: spatial axis. 
        hh :: function that depends on xx.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: lambda function that allows to select the type of spatial derivative
    Output: 
        dt :: time interval
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """       


def cfl_diff_burger(a,dx): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the diffusive term in the Burger's eq. 
    Requires: 
        numpy gradient, abs, and min
    Input: 
        a :: either constant, or array which multiply the right hand side of the Burger's eq.
        x :: spatial axis. 
    Ouput: 
        min(dx/|a|)
    """


def ops_Lax_LL_Add(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    """
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Additive Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
         step_adv_burgers
         cfl_adv_burger
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        b  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def ops_Lax_LL_Lie(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    """
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
         step_adv_burgers
         cfl_adv_burger
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        b  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def ops_Lax_LL_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    """
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme. Both steps are 
    with a Lax method. 

    Requires: 
         step_adv_burgers
         cfl_adv_burger
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        b  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """



def osp_Lax_LH_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    """
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Strang Operator Splitting scheme. One step is with a Lax method 
    and the second step is the Hyman predictor-corrector scheme. 
    Requires: 
         step_adv_burgers
         cfl_adv_burger
         numpy.pad for boundaries. 
    Input: 
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        b  :: either constant, or array which multiply the right hand side of the Burger's eq.
        cfl_cut:: constant value to limit dt from cfl_adv_burger. 
        ddx :: Lambda function allows to change the space derivative function. 
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information. 
    Output: 
        t :: time 1D array
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """ 


def step_diff_burgers(xx, hh, a, **kwargs): 
    """
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that 
    depends on xx. 
    Input:    
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
    Output: 
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """    



def NR_f(xx, un, uo, a, dt, **kwargs): 
    """
    NR F function. 
    Input:    
        xx :: spatial axis. 
        un :: function that depends on xx.
        uo :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
    Output: 
        F :: function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    


def jacobian(xx, un, a, dt, **kwargs): 
    """
    Jacobian of the F function. 
    Input:    
        xx :: spatial axis. 
        un :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
    Output: 
        J :: Jacobian F_j'(u^{n+1}{k})
    """    


def Newton_Raphson(xx, hh, a, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 
    Input:    
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
        toll:: constant of the error limit
        ncount: maximum number of iterations
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information.         
    Output: 
        t :: array of time. 
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
        errt :: error for each timestep
        countt :: number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian(xx, ug, a, dt) # Jacobian 
            ff1=NR_f(xx, ug, uo, a, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            count+=1
            err = np.max(np.abs(un-ug)) # error
            errt[it]=err
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt


def NR_f_u(xx, un, uo, dt, **kwargs): 
    """
    NR F function. 
    Input:    
        xx :: spatial axis. 
        un :: function that depends on xx.
        uo :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
    Output: 
        F :: function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    


def jacobian_u(xx, un, dt, **kwargs): 
    """
    Jacobian of the F function. 
    Input:    
        xx :: spatial axis. 
        un :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
    Output: 
        J :: Jacobian F_j'(u^{n+1}{k})
    """    

def Newton_Raphson_u(xx, hh, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 
    Input:    
        xx :: spatial axis. 
        hh :: function that depends on xx.
        a  :: either constant, or array which multiply the right hand side of the Burger's eq.
        dt :: constant: time interval
        toll:: constant of the error limit
        ncount: maximum number of iterations
        bnd_type:: String. It allows to select the type of boundaries 
        bnd_limits:: Array of two integer elements. The number of pixels that
                will need to be updated with the boundary information.         
    Output: 
        t :: array of time. 
        unnt :: Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
        errt :: error for each timestep
        countt :: number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian_u(xx, ug, dt) # Jacobian 
            ff1=NR_f_u(xx, ug, uo, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            count+=1
            err = np.max(np.abs(un-ug)) # error
            errt[it]=err
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt

def hyman(xx, f, dth, a, fold=None, dtold=None,
        cfl_cut=0.8, ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if (np.any(fold) == None):
        firstit=False
        fold = np.copy(f)
        f = (np.roll(f,1)+np.roll(f,-1))/2.0 + u1_temp * dth 
        dtold=dth

    else:
        ratio = dth/dtold
        a1 = ratio**2
        b1 =  dth*(1.0+ratio   )
        a2 =  2.*(1.0+ratio    )/(2.0+3.0*ratio)
        b2 =  dth*(1.0+ratio**2)/(2.0+3.0*ratio)
        c2 =  dth*(1.0+ratio   )/(2.0+3.0*ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)
        
        if bnd_limits[1]>0: 
            u1_c =  f[bnd_limits[0]:-bnd_limits[1]]
        else: 
            u1_c = f[bnd_limits[0]:]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1]>0: 
        u1_c = f[bnd_limits[0]:-bnd_limits[1]]
    else: 
        u1_c = f[bnd_limits[0]:]
    f = np.pad(u1_c, bnd_limits, bnd_type)
    
    dtold=dth

    return f, fold, dtold


def hyman_corr(f, fsav, dfdt, c2):

    return  fsav  + c2* dfdt


def hyman_pred(f, fold, dfdt, a1, b1, a2, b2): 

    fsav = np.copy(f)
    tempvar = f + a1*(fold-f) + b1*dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2*(fsav-tempvar) + b2*dfdt    
    f = tempvar
    
    return f, fold, fsav
