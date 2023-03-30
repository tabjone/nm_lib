#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import builtin modules
import os

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt 

def deriv_dnw(xx, hh, **kwargs):
    """
    Returns the non-centered 1st order forward derivative of hh respect to xx. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The non-centered 1st order forward derivative of hh respect to xx. 
        Last point is ill calculated. 
    """
    if "ddx_order" in kwargs:
        order = kwargs["ddx_order"]
    else:
        order = 1

    dx = np.roll(xx, -1) - xx
    
    if order == 1:
        return (np.roll(hh, -1) - hh)/dx
    
    elif order == 2:
        return (-np.roll(hh, -2) + 4 * np.roll(hh, -1) - hh)/dx
    
    else:
        raise ValueError('Order not implemented')


def deriv_upw(xx, hh, **kwargs):
    r"""
    returns the upwind 2nd order derivative of hh respect to xx. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    ------- 
    `array`
        The upwind 2nd order derivative of hh respect to xx. First 
        grid point is ill calculated. 
    """
    if "ddx_order" in kwargs:
        order = kwargs["ddx_order"]
    else:
        order = 1

    dx = xx - np.roll(xx, +1)

    if order == 1:
        return (hh - np.roll(hh, +1))/dx
    elif order == 2:
        return (3*hh - 4*np.roll(hh,+1) + np.roll(hh, +2))/(2*dx)
    else:
        raise ValueError('Order not implemented')
        
    
def deriv_cent(xx, hh, **kwargs):
    r"""
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """
    if "ddx_order" in kwargs:
        order = kwargs["ddx_order"]
    else:
        order = 2

    dx = np.roll(xx, -1) - xx

    if order == 2:
        return (np.roll(hh, -1) - np.roll(hh, 1))/(2*dx)

    elif order == 4:
        return (np.roll(hh, +2) - 8 * np.roll(hh, +1) + 8 * np.roll(hh, -1) - np.roll(hh, -2)) / (12*dx)
    else:
        raise ValueError('Order not implemented')


def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function 

    Parameters 
    ----------
    hh : `array`
        Function that depends on xx. 
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh. 
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array` 
        The order of convergence.  
    """
   
def step_adv_burgers(xx, hh, a, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 

    Requires 
    ---------- 
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default clf_cut=0.98. 
    ddx : `lambda function`
        Allows to select the type of spatial derivative. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array` 
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    dt = cfl_cut * cfl_adv_burger(a, xx)

    return dt, - a * ddx(xx, hh, **kwargs)

def cfl_adv_burger(a,x): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the advective term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    ------- 
    `float`
        min(dx/|a|)
    """
    return np.min(np.gradient(x)/np.abs(a))


def evolv_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).  
    bnd_type : `string`
        Allows to select the type of boundaries. 
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1].

    Returns
    ------- 
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    dt = cfl_cut * cfl_adv_burger(a, xx)

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_adv_burgers(xx, unnt[i,:], a, ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = unnt[i,:] + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt

    return tt, unnt

def evolv_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger. 
        By default 0.98.
    ddx : `lambda function` 
        Allows to change the space derivative function. 
    bnd_type : `string` 
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array` 
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_uadv_burgers(xx, unnt[i,:], ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = unnt[i,:] + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt

    return tt, unnt
    

def step_uadv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.  

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """   
    dt = cfl_cut * cfl_adv_burger(hh, xx)

    return dt, - hh * ddx(xx, hh, **kwargs)    

def evolv_Lax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_uadv_burgers(xx, unnt[i,:], ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = 0.5 * (np.roll(hh, -1) + np.roll(hh, +1)) + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt
    return tt, unnt


def evolv_Lax_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_adv_burgers(xx, hh,a, ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = 0.5 * (np.roll(hh, -1) + np.roll(hh, +1)) + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt
    return tt, unnt

def step_Rie_uadv_burgers(xx, hh, clf_cut = 0.98,
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs):
    r"""
    Computes the timestep and the right hand side of the Burger's eq
    for the Riemann problem. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    clf_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98

    Returns
    -------
    dt : `float`
        Timestep
    rhs : `array`
        Right hand side of the Burger's eq
    """
    uL = hh
    uR = np.roll(hh, -1)
    fL = 1/2 * uL**2
    fR = 1/2 * uR**2

    propSpeedL = np.abs(uL)
    propSpeedR = np.abs(uR)

    propSpeed = np.max([propSpeedL, propSpeedR], axis=0)

    #this is grid shifted +1/2 to the right
    interfaceFlux = 1/2 * (fL + fR) - 1/2 * propSpeed * (uR - uL)

    #compute dt
    dt = cfl_adv_burger(propSpeed, xx)
    dx = np.roll(xx, -1) - xx

    rhs = - (interfaceFlux - np.roll(interfaceFlux, 1))/dx
    return dt, rhs


def evolv_Rie_uadv_burgers(xx, hh, nt, cfl_cut = 0.98,
        ddx = lambda x,y: deriv_dnw(x, y),
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    """

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0


    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_Rie_uadv_burgers(xx, unnt[i,:], ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = unnt[i,:] + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt

    return tt, unnt





def ops_Lax_LL_Add(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Additive Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt1, rhs_u = step_adv_burgers(xx, unnt[i,:], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        dt2, rhs_v = step_adv_burgers(xx, unnt[i,:], b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        dt = np.min([dt1, dt2])
        #forwarding in time
        uu = 0.5 * (np.roll(unnt[i,:], -1) + np.roll(unnt[i,:], +1)) + rhs_u * dt
        vv = 0.5 * (np.roll(unnt[i,:], -1) + np.roll(unnt[i,:], +1)) + rhs_v * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            uu = uu[bnd_limits[0]:-bnd_limits[1]]
            vv = vv[bnd_limits[0]:-bnd_limits[1]]
        else:
            uu = uu[bnd_limits[0]:]
            vv = vv[bnd_limits[0]:]
        #padding
        uu = np.pad(uu, pad_width=bnd_limits ,mode=bnd_type)
        vv = np.pad(vv, pad_width=bnd_limits ,mode=bnd_type)


        unnt[i+1,:] = uu + vv - unnt[i,:]
        tt[i+1] = tt[i] + dt
    return tt, unnt

def ops_Lax_LL_Lie(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0



    for i in range(0,nt-1):
        dt1 = cfl_adv_burger(a, xx)
        dt2 = cfl_adv_burger(b, xx)
        dt = np.min([dt1, dt2])
        #getting timestep and rhs of Burgers eq
        _, rhs_u = step_adv_burgers(xx, unnt[i,:], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        #forwarding u in time
        uu = 0.5 * (np.roll(unnt[i,:], -1) + np.roll(unnt[i,:], +1)) + rhs_u * dt

        #remove ill calculated points
        if bnd_limits[1] != 0:
            uu = uu[bnd_limits[0]:-bnd_limits[1]]
        else:
            uu = uu[bnd_limits[0]:]
        
        #padding
        uu = np.pad(uu, pad_width=bnd_limits ,mode=bnd_type)

        #spacial derivative of v with uu as input
        _, rhs_v = step_adv_burgers(xx, uu, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        
        #forwarding v in time
        vv = 0.5 * (np.roll(uu, -1) + np.roll(uu, +1)) + rhs_v * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            vv = vv[bnd_limits[0]:-bnd_limits[1]]
        else:
            vv = vv[bnd_limits[0]:]
        #padding
        vv = np.pad(vv, pad_width=bnd_limits ,mode=bnd_type)

        unnt[i+1,:] = vv
        tt[i+1] = tt[i] + dt
    return tt, unnt



def ops_Lax_LL_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme. Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)` 
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0


    for i in range(0,nt-1):
        dt1 = cfl_adv_burger(a, xx)
        dt2 = cfl_adv_burger(b, xx)
        dt = np.min([dt1, dt2])
        #getting timestep and rhs of Burgers eq
        _, rhs_u = step_adv_burgers(xx, unnt[i,:], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        #Forwarding u a half step in time
        uu = 0.5 * (np.roll(unnt[i,:], -1) + np.roll(unnt[i,:], +1)) + rhs_u * dt/2
        
        #remove ill calculated points
        if bnd_limits[1] != 0:
            uu = uu[bnd_limits[0]:-bnd_limits[1]]
        else:
            uu = uu[bnd_limits[0]:]
        #padding
        uu = np.pad(uu, pad_width=bnd_limits ,mode=bnd_type)

        #spacial derivative of v with uu as input
        _, rhs_v = step_adv_burgers(xx, uu, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        
        #full step vv in time
        vv = 0.5 * (np.roll(uu, -1) + np.roll(uu, +1)) + rhs_v * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            vv = vv[bnd_limits[0]:-bnd_limits[1]]
        else:
            vv = vv[bnd_limits[0]:]
        #padding
        vv = np.pad(vv, pad_width=bnd_limits ,mode=bnd_type)

        #spacial derivative of ww with vv as input
        _, rhs_w = step_adv_burgers(xx, vv, a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        #forwarding w in time, half timestep
        ww = 0.5 * (np.roll(vv, -1) + np.roll(vv, +1)) + rhs_w * dt/2

        unnt[i+1,:] = ww
        tt[i+1] = tt[i] + dt
    return tt, unnt



def osp_Lax_LH_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Strang Operator Splitting scheme. One step is with a Lax method 
    and the second step is the Hyman predictor-corrector scheme. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0



    for i in range(0,nt-1):
        dt1 = cfl_adv_burger(a, xx)
        dt2 = cfl_adv_burger(b, xx)
        dt = np.min([dt1, dt2])

        #getting timestep and rhs of Burgers eq
        _, rhs_u = step_adv_burgers(xx, unnt[i,:], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        #Forwarding u a half step in time
        uu = 0.5 * (np.roll(unnt[i,:], -1) + np.roll(unnt[i,:], +1)) + rhs_u * dt/2
        
        #remove ill calculated points
        if bnd_limits[1] != 0:
            uu = uu[bnd_limits[0]:-bnd_limits[1]]
        else:
            uu = uu[bnd_limits[0]:]
        #padding
        uu = np.pad(uu, pad_width=bnd_limits ,mode=bnd_type)

        #spacial derivative of v with uu as input
        _, rhs_v = step_adv_burgers(xx, uu, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        
        #full step vv in time
        if i==0:
            vv, uo, dt_v = hyman(xx, uu, dt, b, cfl_cut=cfl_cut, ddx=ddx,
                                       bnd_limits=bnd_limits)
        else:
            vv, uo, dt_v =  hyman(xx, uu, dt, b, cfl_cut=cfl_cut, ddx=ddx, bnd_limits=bnd_limits,
                                    fold=uo, dtold=dt_v)

        #remove ill calculated points
        if bnd_limits[1] != 0:
            vv = vv[bnd_limits[0]:-bnd_limits[1]]
        else:
            vv = vv[bnd_limits[0]:]
        #padding
        vv = np.pad(vv, pad_width=bnd_limits ,mode=bnd_type)

        #spacial derivative of ww with vv as input
        _, rhs_w = step_adv_burgers(xx, vv, a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        #forwarding w in time, half timestep
        ww = 0.5 * (np.roll(vv, -1) + np.roll(vv, +1)) + rhs_w * dt/2

        unnt[i+1,:] = ww
        tt[i+1] = tt[i] + dt
    return tt, unnt


def evolv_diff_burgers(xx, hh,nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    Bla bla bla 
    """

    tt = np.zeros(nt)
    unnt = np.zeros((nt, len(xx)))

    #setting initial values
    unnt[0,:] = hh
    tt[0] = 0

    for i in range(0,nt-1):
        #getting timestep and rhs of Burgers eq
        dt, rhs = step_diff_burgers(xx, unnt[i,:], a, ddx=ddx, cfl_cut=cfl_cut, **kwargs)
        #forwarding in time
        hh = unnt[i,:] + rhs * dt
   
        #remove ill calculated points
        if bnd_limits[1] != 0:
            hh = hh[bnd_limits[0]:-bnd_limits[1]]
        else:
            hh = hh[bnd_limits[0]:]
        #padding
        hh = np.pad(hh, pad_width=bnd_limits ,mode=bnd_type)
        unnt[i+1,:] = hh
        tt[i+1] = tt[i] + dt

    return tt, unnt

def step_diff_burgers(xx, hh, a, ddx = lambda x,y: deriv_cent(x, y), cfl_cut=0.98, **kwargs): 
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that 
    depends on xx. 
    
    Parameters
    ----------    
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    #first up, then down
    dt = cfl_cut * cfl_diff_burger(a, xx)
    return dt,  a * deriv_dnw(xx, deriv_upw(xx, hh, **kwargs), **kwargs)

def cfl_diff_burger(a,x): 
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the diffusive term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    grad_x = np.gradient(x)
    return np.min(grad_x*grad_x/(4*np.abs(a)))


def NR_f(xx, un, uo, a, dt, **kwargs): 
    r"""
    NR F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """
    dx = xx[1] - xx[0]
    #F function
    return un - uo - a * (np.roll(un, -1) - 2 * un + np.roll(un, +1)) * dt


def jacobian(xx, un, a, dt, **kwargs): 
    r"""
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """
    #Derivative of F function
    dx = xx[1] - xx[0]
    jac = np.zeros((np.size(xx), np.size(xx)))
    for ix in range(np.size(xx)):
        jac[ix, ix] = 1 + dt * 2 * a/(dx**2)
        if ix < np.size(xx) - 1:
            jac[ix, ix+1] = - dt * a/(dx**2)
        if ix > 1:
            jac[ix, ix-1] = - dt * a/(dx**2)
    return jac



def Newton_Raphson(xx, hh, a, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    NR scheme for the burgers equation. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt):
        #u(x) for some given timestep 
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

            # error: 
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) # error
            #err = np.max(np.abs(un-ug))
            errt[it]=err

            # Number of iterations
            count+=1
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
    r"""
    NR F function.

    Parameters
    ----------  
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """
    dx = xx[1] - xx[0]
    #F function
    return un - uo - uo * (np.roll(un, -1) - 2 * un + np.roll(un, +1)) * dt



def jacobian_u(xx, un, dt, **kwargs): 
    """
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """    
    #Derivative of F function
    dx = xx[1] - xx[0]
    jac = np.zeros((np.size(xx), np.size(xx)))
    for ix in range(np.size(xx)):
        jac[ix, ix] = 1 + dt * 2 * un[ix]/(dx**2)
        if ix < np.size(xx) - 1:
            jac[ix, ix+1] = - dt * un[ix]/(dx**2)
        if ix > 1:
            jac[ix, ix-1] = - dt * un[ix]/(dx**2)
    return jac


def Newton_Raphson_u(xx, hh, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float` 
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]        

    Returns
    -------
    t : `array`
        Time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `array(int)` 
        Number iterations for each timestep
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

            # error
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) 
            errt[it]=err

            # Number of iterations
            count+=1
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

def taui_sts(nu, niter, iiter): 
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------   
    nu : `float`
        Coefficient, between (0,1).
    niter : `int` 
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float` 
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """

def evol_sts(xx, hh, nt,  a, cfl_cut = 0.45, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], nu=0.9, n_sts=10): 
    """
    Evolution of the STS method. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.45
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations. 
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


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
