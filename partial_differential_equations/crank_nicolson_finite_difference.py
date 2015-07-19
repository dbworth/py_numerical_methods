"""
Solution to the temperature PDE for a Gardon Gauge[1]
using the Crank-Nicolson finite difference method

David Butterworth
University of Queensland


[1] "An Instrument for the Direct Measurement of Intense Thermal Radiation" by Robert Gardon, 1953

"""

#------------------------------------------------------------------------------#

import math # pi
import numpy as np # zeros, linspace, linalg
import pylab

# for 3D surf plots:
from mpl_toolkits.mplot3d import axes3d,Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------#

NO_PLOT   = 0
PLOT_2D   = 1
PLOT_2D_2 = 2
PLOT_3D   = 3
NO_SAVE   = 0
SAVE_PLOT = 1

#------------------------------------------------------------------------------#

def simulateCN(d, s, q, Tstop, I, test_number, plot_type, save_figure):
    """
    Simulate behavior of heating the gauge using C-N scheme.
        
    d       Disk Diameter (cm)
    s       Foil Thickness (cm)
    q       Thermal Radiation Intensity (cal/cm^2/sec)
    Tstop   Simulation time (sec)
    I       Number of spatial steps

    test_number     The disk number, used for labelling plots/images
    plot_type = 0   Don't display a plot for this simulation
                1   2D plot of temperature at center of disk
                2   2D plot of final temperature across radius
                3   3D plot of temperature history across radius
    """

    # Constants for a Constantan-Copper Gardon Gauge
    c = 0.094 # Specific Heat (gr.cal/degC)
    p = 8.9   # Density (gr/cm^3)
    k0 = 0.052 # Base thermal conductivity at T=0 (cal/cm/sec/degC)
    alpha = 0.0023 # Thermal Conductivity Coefficient (1/degC)

    # Variables
    #d = d # Foil Diameter (cm)
    S = s # Foil Thickness (cm)
    R = d / 2  # Radius (cm)

    # Spatial discretization 
    a = 0.0
    b = R
    N = I # N = number of segments, N+1 rows in matrix system
          # 100 is minimum of reasonable result, but 500 increases accuray by a few degrees
    dr = (b-a)/N # split radius into N pieces

    # Time discretization (should be limited for stability)
    t_final = Tstop  #0.250
    dt = 0.001 # Minimum dt = 0.01 for disk 1, but need 0.001 for other disks, 
               # which should be bigger than FTCS can do, 
              # and in fact smaller values are not required for CH method

    # Vector to store temperature values T at current time t
    T_t = np.zeros((N+1,), np.float64)
    # ...and next time t+1
    T_tp1 = np.zeros((N+1,), np.float64)

    # Matrix to store LHS of linear system,
    # these could be fixed if equations were only a function of radius r,
    # but they're also a function of temperature T, which means at 
    # each time step the LHS will change.
    LHS_mat = np.zeros((N+1,N+1),np.float64)
    # ...and vector to store RHS of system, which will also
    # vary with r and T.
    RHS_vec = np.zeros((N+1,),np.float64)


    # Compute the thermal coefficient k,
    # alpha is constant, but temperature T is the
    # current temperature T_t at some radius r.
    calculate_k = lambda T: k0 * (1 + alpha * T)

    # Compute the co-efficients that form the LHS and RHS of
    # the system, these vary spatially with radius r, and/or
    # vary in time as k changes
    calculate_A = lambda r: 2.0*math.pi*r*dr*S*c*p # radius varies with spatial position
    calculate_B = lambda r: 2.0*math.pi*r*dr*q # r
    calculate_C = lambda r,k: k*2.0*math.pi*S*((r + dr) - r) # r and k vary
    calculate_D = lambda r,k: dr*k*2.0*math.pi*S*(r + dr) # r and k vary


    # Create vector of radius values from r=0 to r=R,
    # for plotting data
    temp = np.linspace(a, b, N+1)
    r_vec = np.zeros((N+1,),np.float64)
    for i in range(np.size(r_vec)): r_vec[i] = temp[i] # copy linspace into vector


    # Calculate number of time steps from t=0 to t=t_final
    num_t_steps = t_final / dt
    #print "num_t_steps =", num_t_steps
    if num_t_steps == int(num_t_steps): # If dt divides in perfectly, the number
        num_t_steps = int(num_t_steps)  # of time steps is a whole number,
    else:                                # else, dt is not a divisor, so round time steps
        num_t_steps = int(num_t_steps)+1 # up to next whole number.


    # Create vector of time values from t=0 to t=t_final,
    # for plotting data
    temp2 = np.linspace(0.0, t_final, num_t_steps+1)
    t_vec = np.zeros((num_t_steps+1,),np.float64)
    for i in range(np.size(t_vec)): t_vec[i] = temp2[i] # copy linspace into vector


    # Create matrix for temperature results, for 3D plotting
    # rows = time
    # cols = radius
    T_result = np.zeros(( np.size(t_vec), np.size(r_vec) ),np.float64)

    # Expand vector of radius values into a matrix, for 3D plotting
    r_mat = np.zeros(( np.size(t_vec), np.size(r_vec) ),np.float64)
    for row in range(np.size(t_vec)): 
       for col in range(np.size(r_vec)): 
           r_mat[row,col] = r_vec[col] # each row has the same radius values

    # Expand vector of time values into a matrix, for 3D plotting
    t_mat = np.zeros(( np.size(t_vec), np.size(r_vec) ),np.float64)
    for row in range(np.size(t_vec)): 
       for col in range(np.size(r_vec)): 
           t_mat[row,col] = t_vec[row] # each column has the same time values

    # For storing temperature T history at center of disk
    T_at_center = np.zeros((np.size(t_vec),),np.float64)

    # For storing final temperature T over entire radius of disk
    final_T_over_radius = np.zeros((np.size(r_vec),),np.float64)


    # Integrate the PDE over time t
    ti = 1 # counter, number of time steps, skip t=0 where temperature T=0
    t = 0.0 # for controlling the loop
    while t <= t_final:
        t += dt

        # At each time step re-build the linear system,
        # as both the LHS and RHS have variables that change with time
        for row in range(N+1):
            col = row # for a tri-diagonal system

            r = row * dr # current radius

            # k is a function of the temperature T at this position
            T = T_t[row]
            k = calculate_k(T) # T
            #k = 0.06 # test, for fixed k

            # For this spatial position r, and time t,
            # calculate the values of the coefficients in the system 
            A = calculate_A(r)
            B = calculate_B(r)
            C = calculate_C(r,k)
            D = calculate_D(r,k)

            # LHS matrix:

            # first row
            if row == 0:
                #LHS_mat[row,col-1] = C/(4.0*dr) - D/(2.0*dr*dr) # col-1 does not exist, this is applied to col+1 below
                LHS_mat[row,col]   = A/dt + D/(dr*dr)
                 # note the 2nd C and D are for col-1 which is reflected at r=0
                LHS_mat[row,col+1] = (-1.0*C)/(4.0*dr) - D/(2.0*dr*dr) + C/(4.0*dr) - D/(2.0*dr*dr) 
            # interior rows
            if (row != 0) and (row != N):
                LHS_mat[row,col-1] = C/(4.0*dr) - D/(2.0*dr*dr)
                LHS_mat[row,col]   = A/dt + D/(dr*dr)
                LHS_mat[row,col+1] = (-1.0*C)/(4.0*dr) - D/(2.0*dr*dr)
            # last row, for the boundary condition
            if row == N:
                LHS_mat[row,col]   = 1.0


            # RHS vector

            # first entry
            if row == 0:
                # note the first (blah)*row+1 term is for row-1, which is reflected at r=0
                RHS_vec[row] = B + ( (-1.0*C)/(4.0*dr) + D/(2.0*dr*dr) )*T_t[row+1] + ( A/dt - D/(dr*dr) )*T_t[row] + ( C/(4.0*dr) + D/(2.0*dr*dr) )*T_t[row+1]
            # interior entries
            if (row != 0) and (row != N):
                RHS_vec[row] = B + ( (-1.0*C)/(4.0*dr) + D/(2.0*dr*dr) )*T_t[row-1] + ( A/dt - D/(dr*dr) )*T_t[row] + ( C/(4.0*dr) + D/(2.0*dr*dr) )*T_t[row+1] 
            # last entry, the boundary condition
            if row == N:
                RHS_vec[row] = 0.0 # boundary condition, temp T=0 at r=R


        # Solve system to find next temperatures T_t+1 at each radius
        T_tp1 = np.linalg.solve(LHS_mat, RHS_vec)

        # Temperature T_t+1 becomes T_t for next step
        T_t = T_tp1.copy()

        # Sometimes if dt was a divisor, matrix might be one size too small
        if i > num_t_steps: print "\nwarning: i > num_t_steps \n" 

        # Save data for later plotting
        T_at_center[ti] = T_tp1[0] # store temperature T at r=0 (center of disk)

        # Store complete temperature history
        for ri in range(np.size(T_tp1)):
            T_result[ti,ri] = T_tp1[ri] # rows = time, cols = radius

        ti = ti+1 # increment iterator
    #:end while

    # print final temperature T across radius
    print "T_tp1 =", T_tp1
    # Print temperature history at center of disk
    #print "T_at_center =", T_at_center

    # Retrieve final temperature T values across the radius r of the disc
    for i in range(np.size(T_tp1)):
        final_T_over_radius[i] = T_tp1[i]

    if plot_type == PLOT_2D:
        # 2D plot of temperature history at center of disk
        pylab.figure()
        pylab.plot(t_vec, T_at_center, 'k-', label='Center of Disk') # plot temperature T at center
        #pylab.axis( [0.0, t_final, 0.0, 350] ) # Set the t,T (x,y) axes range
        pylab.axis( [0.0, t_final, 0.0, 60] ) 
        pylab.title("Disk %d:  Time (t)  vs.  Temperature (T)  (q = %d)" % (test_number, q))
        pylab.legend(loc='best')
        pylab.grid()
        pylab.xlabel("Time (sec)")
        pylab.ylabel(u"Temperature (\N{DEGREE SIGN}C)")
        if save_figure == SAVE_PLOT:
            pylab.savefig('disk%d-2dplot-center.png' % test_number) # save plot
        pylab.show()

    if plot_type == PLOT_2D_2:
        # 2D plot of final temperature across the radius
        pylab.figure()
        pylab.plot(r_vec, final_T_over_radius, 'k--', label='Final temperature') # plot final temperature T across the radius
        pylab.axis( [0.0, R, 0.0, 350] ) # Set the t,T (x,y) axes range
        pylab.title("Disk %d:  Radius (r)  vs.  Temperature (T)  (q = %d)" % (test_number, q))
        pylab.legend(loc='best')
        pylab.grid()
        pylab.xlabel("Radius (cm)")
        pylab.ylabel(u"Temperature (\N{DEGREE SIGN}C)")
        if save_figure == SAVE_PLOT:
            pylab.savefig('disk%d-2dplot-finaltemp.png' % test_number) # save plot
        pylab.show()

    if plot_type == PLOT_3D:
        # 3D plot of temperature history across radius
        fig = plt.figure()

        #ax = fig.gca(projection='3d') # for newer Python
        ax = Axes3D(fig) # for older Python

        # color map RdBu_r is reversed RdBu, so red indicates higher temp
        surf = ax.plot_surface(t_mat, r_mat, T_result, rstride=1, cstride=1, cmap=cm.RdBu_r,linewidth=0, antialiased=False) # x,y,z
        fig.colorbar(surf, shrink=0.5, aspect=5) # display color to temperature value index
        ax.azim = -120 # set initial viewing angle
        ax.elev = 30
        ax.set_xlabel('time (sec)') # label axes
        ax.set_ylabel('radius (r)')
        ax.set_zlabel(u'temperature (\N{DEGREE SIGN}C)')
        if save_figure == SAVE_PLOT:
            fig.savefig('disk%d-3dplot.png' % test_number) # save image
        plt.show()

    #:end simulateCN()

#------------------------------------------------------------------------------#

print "Solving the Gardon Gauge temperature PDE"

Tstop = 0.250 # simulation time (sec)
I = 500 # number of spatial steps (100 is a good minimum for C-N method)

#
# If you enable the 3D plots, they can take a long time to
# calculate and display the plot!!
#

#print "Disk size 1"
#disk_number = 1
#q = 10 # Incident thermal radiation intensity (cal/cm^2/sec)
#d = 0.315  # disk diameter (cm)
#s = 0.0025 # foil thickness (cm)
#simulateCN(d, s, q, Tstop, I, disk_number, PLOT_3D, SAVE_PLOT)
#simulateCN(d, s, q, Tstop, I, disk_number, PLOT_2D, SAVE_PLOT)
#simulateCN(d, s, q, Tstop, I, disk_number, NO_PLOT, NO_SAVE)


print "Disk size 2"
disk_number = 2
q = 10 # Incident thermal radiation intensity (cal/cm^2/sec)
d = 0.096 # disk diameter (cm)
s = 0.002 # foil thickness (cm)
#simulateCN(d, s, q, Tstop, I, disk_number, PLOT_3D, SAVE_PLOT)
simulateCN(d, s, q, Tstop, I, disk_number, PLOT_2D, SAVE_PLOT)
#simulateCN(d, s, q, Tstop, I, disk_number, NO_PLOT, NO_SAVE)


#print "Disk size 3"
#disk_number = 3
#q = 10 # Incident thermal radiation intensity (cal/cm^2/sec)
#d = 0.254  # disk diameter (cm)
#s = 0.0025 # foil thickness (cm)
#simulateCN(d, s, q, Tstop, I, disk_number, PLOT_3D, SAVE_PLOT)
#simulateCN(d, s, q, Tstop, I, disk_number, PLOT_2D, SAVE_PLOT)
#simulateCN(d, s, q, Tstop, I, disk_number, NO_PLOT, NO_SAVE)

