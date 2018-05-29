import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.interpolate import BSpline, splprep, splev
import scipy.integrate as integrate
import time
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import pickle

def planar_rigid_body_transformation_on_path(poses, t=None, phi=None, Trans=None):
    if Trans is None:
        assert(t is not None)
        assert(phi is not None)
        Trans = np.array([[np.cos(phi), -np.sin(phi), t[0]],
                      [np.sin(phi), np.cos(phi), t[1]],
                      [0, 0, 1]])

    """ poses must be in shape [batch, num, 3]
        t is translation vector = [tx, ty]
        phi is rotation angle
    """
    batch, num, dim = poses.shape
    poses_ = copy.deepcopy(poses)
    i = 0
    for traj in poses_:
        temp = copy.deepcopy(traj[:,2])
        traj[:,2] = 1
        X_new = np.matmul(Trans, traj.T)
        poses_[i] = X_new.T
        poses_[i,:,2] = temp
        i += 1
    return poses_

def planar_scaling_on_path(poses, s):
    """ poses must be in shape [batch, num, 3]
        t is translation vector = [tx, ty]
        phi is rotation angle
    """
    batch, num, dim = poses.shape
    Trans = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]])
    i = 0
    poses_ = copy.deepcopy(poses)
    for traj in poses_:
        temp = copy.deepcopy(traj[:,2])
        traj[:,2] = 1
        X_new = np.matmul(Trans, traj.T)
        poses_[i] = X_new.T
        poses_[i,:,2] = temp
        i += 1
    return poses_

def spatial_rigid_body_transformation_on_3d_curve(poses, R, t):
    """ poses must be in shape [batch, num, 3]
        R is rotation matrix
        t is translation vector = [tx, ty]
    """
    pass

def signature(x, y, angle, ax=None, ax2=None, ax3=None, label='', steps=100, output_steps = 50):
    """ Obtains the invariant signature of motion
    """
    diff_angle = [0]
    for i in range(1,len(angle)):
            diff_angle.append(angle[i] - angle[i-1])
            if np.abs(diff_angle[i]) >= np.pi:
                # Calculate absolute diff
                delta = 2*np.pi - np.abs(diff_angle[i])
                diff_angle[i] = - delta*np.sign(diff_angle[i])

    # Normalizing angle
    angle = np.cumsum(diff_angle)
    # Step 1
    """ 1. This step covers any pre-processing required to de-noise and convert
            the input curve to a parametric representation such that the curvature
            can be readily and robustly estimated for the curves. For example if
            the input is a digitized point set, one option is to approximate with a
            parametric cubic B-spline curve.
    """
    #if ax2 is not None:
    #    ax2.plot(x, y, '*', ms=4, label=label+' : B-Spline')
    #    ax2.plot(x, y, '*', ms=4, label=label)
    # Uniform Arc Length Parametrization
    u_arc = [0]
    dist = 0
    x_mean = np.mean(np.max(x)+np.min(x))
    y_mean = np.mean(np.max(y)+np.min(y))
    for i in range(1,len(x)):
        dist = dist + np.power((x[i] - x_mean)**2 + (y[i] - y_mean)**2, 0.5)
        u_arc.append(dist)
    u_arc = u_arc/u_arc[-1]

    # Preparing Spline of fitting Path
    #tck, _ = splprep(x=[x, y, angle], u=u_arc, k=3, s=0)
    tck, _ = splprep(x=[x, y, angle], k=3, s=0)
    # Evaluating Spline for many points for smoother evaluation
    u = np.arange(0, u_arc[-1], u_arc[-1]/steps)
    x, y, angle = splev(u, tck)
    if ax2 is not None:
        ax2.plot(x, y, '--', ms=2, lw=2, label=label+' : B-Spline')
        ax2.plot(x[0], y[0], '*', ms=10, label=label+': Start Point')
        #ax2.plot(x, y, 'o', ms=2, label='B-spline')

    """
        2. Sample the curvature of the curve at equal arc length intervals. Figure
             shows curvature K w.r.t. the arc length plot.
    """

    # Velocity
    v = splev(u, tck, der=1)
    # Accelation
    a = splev(u, tck, der=2)
    # Cross Product
    cur_z = - a[0]*v[1] + a[1]*v[0]
    # Curvature
    curvature = cur_z / np.power((v[0]**2 + v[1]**2), 3/2)
    """
        3. Integrate the unsigned curvatures along the curve by summing the absolute
            values of curvatures discretely sampled along the curve. Plot the
            integral of the unsigned curvature K =  |K| w.r.t. the arc length s ;
            see Figure 4(c).
    """
    # Cumulative Integral Calculation
    K = integrate.cumtrapz(np.abs(curvature), u)
    K = np.concatenate((np.array([0]), K))
    if ax is not None:
        ax.plot(K, curvature, 'o', ms=4)
        pass

    """
        4. Compute curvature (k) of the curve at equal-interval-sampled points
            along the integral of unsigned curvature axis (vertical axis in previous
            figure 4(c)) that is, (signed) curvature (k) w.r.t. the integral of unsigned
            curvature (K) plot see Figure 4(d). This is our signature and the core
            of our method. This can also be considered as a novel scale invariant
            parameterization of a curve.
    """
    # Fitting splines through curvature plots
    # Uniform Arc Length Parametrization
    u_cur = [0]
    dist = 0
    for i in range(1,len(K)):
        dist = dist + np.power((K[i] - K[i-1])**2, 0.5)
        u_cur.append(dist)
    u_cur = u_cur/u_cur[-1]

    # Preparing Spline of fitting Path
    tck_cur, _ = splprep(x=[K, curvature, angle], u=u_cur, k=3, s=0)
    # Evaluating Spline for many points for smoother evaluation
    # adaptive step size based on curvature integral
    # for comparing correlation steps = 1/(K[-1]*100)
    u_ = np.arange(0, 1, 1.0/output_steps)
    K_init, curvature_init = K, curvature
    K, curvature, angle = splev(u_, tck_cur)

    if ax is not None:
        ax.plot(K, curvature, '*-', ms=4, lw=2)

    u_1 = np.arange(0, 1, 1.0/K[-1]/100)
    _, path_sign, motion_sign = splev(u_1, tck_cur)
    if ax3 is not None:
        ax3.plot(path_sign, '*-', ms=4, lw=2)

    return {'path_sign': path_sign, 'motion_sign':motion_sign, 'fixed_path_sign':np.array([curvature, K]), 'fixed_motion_sign':np.array([angle, K])}

def normalized_cross_corelation(sign1, sign2, ax=None, ax2=None):
    """ Obtains similarity between two scaled signals of same wavelength
    """
    s1 = sign1['path_sign']
    s2 = sign2['path_sign']

    if len(s2) >= len(s1):
        t = s1
        F = s2
    else:
        t = s2
        F = s1

    if ax is not None:
        ax.plot(t, 'o', ms=5, lw=2, label='Templete')
        ax.plot(F, 'v', ms=5, lw=2, label='Long Curve')

    t_mean = np.mean(t)
    v1 = []
    v2 = []
    tspan = len(t)
    t_diff_square = np.sum(np.power(t - t_mean,2))
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        v1.append(num/np.sqrt(f_diff_square*t_diff_square))
    F = F[::-1]
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum((F[u:u+tspan]-F_mean)*(t - t_mean))
            f_diff_square = np.sum(np.power(F[u:u+tspan]-F_mean, 2))
        v2.append(num/np.sqrt(f_diff_square*t_diff_square))

    if ax2 is not None:
        ax2.plot(v1,'*',ms='4', label='Forward')
        ax2.plot(v2,'-',ms='4', label='Reversed')

    v1 = np.square(v1)
    v2 = np.square(v2)
    v = [np.max(v1), np.max(v2)]
    v_ = [v1, v2]
    score = np.max(v)
    order = np.argmax(v)
    offset = np.argmax(v_[order])
    output = {'score': score, 'order': order, 'offset': offset}
    if np.isnan(score):
        return {'score': -1, 'order': order, 'offset': offset}
    else:
        return output

def motion_cross_corelation(sign1, sign2, ax=None, ax2=None):
    """ Obtains similarity between two scaled signals of same wavelength
    """
    s1 = sign1['motion_sign']
    s2 = sign2['motion_sign']

    if len(s2) >= len(s1):
        t = s1
        F = s2
    else:
        t = s2
        F = s1
    if ax is not None:
        ax.plot(t, 'v', ms=5, lw=2, label='Templete')
        ax.plot(F, '-', ms=5, lw=2, label='Long Curve')

    t_mean = np.mean(t)
    v1 = []
    v2 = []
    tspan = len(t)
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        v1.append(num)

    F = F[::-1]
    for u in range(len(F) - len(t) + 1):
        if(u==0):
            F_sum = np.sum(F[u:u+tspan])
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        else:
            F_sum = F_sum + F[u+tspan-1] - F[u]
            F_mean = F_sum/tspan
            num = np.sum(np.power((F[u:u+tspan]-F_mean)-(t - t_mean),2))
        v2.append(num)

    if ax2 is not None:
        ax2.plot(v1,'--',ms='4', label='Forward')
        ax2.plot(v2,'-',ms='4', label='Reverse')
    v = [np.min(v1), np.min(v2)]
    v_ = [v1, v2]
    distance = np.min(v)
    order = np.argmin(v)
    offset = np.argmin(v_[order])
    output = {'distance': distance, 'order': order, 'offset': offset}
    return output

def transform_to_planar_dual_qt(poses):
    x = poses[:,:,0]
    y = poses[:,:,1]
    theta = poses[:,:,2]
    z3 = np.sin(theta / 2);
    z4 = np.cos(theta / 2);
    z1 = 0.5 * (x * z3 - y * z4);
    z2 = 0.5 * (x * z4 + y * z3);
    return np.transpose((z1, z2, z3, z4), [1, 2, 0])

def ff_descriptor(Ks, max_harmonics=5, T=0.01):
    """ input: signature Ks should be in shape =  [batch, 101]
        output: fourier discriptors upto max_harmonics
    """
    f_descriptors = np.zeros((len(Ks), max_harmonics),dtype=np.float32)
    i = 0
    for y in Ks:
        n = len(y) # length of the signal
        k = np.arange(n)
        frq = k/T # two sides frequency range
        frq = frq[range(n/2)] # one side frequency range

        Y = np.fft.fft(y)/n # fft computing and normalization
        Y = Y[range(n/2)]
        f_descriptors[i,:] = np.abs(Y[:max_harmonics])
        i += 1
    return f_descriptors

def get_pca_transform(qx, qy, ax=None, label=''):
    """ Performs the PCA
        Return transformation matrix
    """
    cx = np.mean(qx)
    cy = np.mean(qy)
    covar_xx = np.sum((qx - cx)*(qx - cx))/len(qx)
    covar_xy = np.sum((qx - cx)*(qy - cy))/len(qx)
    covar_yx = np.sum((qy - cy)*(qx - cx))/len(qx)
    covar_yy = np.sum((qy - cy)*(qy - cy))/len(qx)
    covar = np.array([[covar_xx, covar_xy],[covar_yx, covar_yy]])
    eig_val, eig_vec= np.linalg.eig(covar)

    # Inclination of major principal axis w.r.t. x axis
    if eig_val[0] > eig_val[1]:
        phi= np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        phi= np.arctan2(eig_vec[1,1], eig_vec[0,1])
    # Transformation matrix T
    trans = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0, 1]])
    rot = np.array([[np.cos(phi), np.sin(phi), 0],
                  [-np.sin(phi), np.cos(phi), 0],
                  [0, 0, 1]])
    Trans = np.matmul(rot,trans)
    if ax is not None:
        ax.plot(qx, qy, '*',ms=4, label=label)
        phi_m = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        e = Ellipse(xy=[cx, cy], width=eig_val[0], height=eig_val[1], angle=phi_m*180/np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
    max_eig = 1/np.sqrt(np.max(eig_val))
    return Trans, ax, max_eig

def transform_poses(x,y,theta, ax=None, label=''):
    """ Translates and rotates path part of the poses along
        Principal Directions
        input: x shape :[num, 3]
        Return: Poses => (x_transformed, y_transformed, theta_original)
    """
    #T, ax, scale_factor = get_pca_transform(x, y, ax=ax, label=label + ' : Original')
    pose = np.transpose((x, y, theta),[1,0])
    #pose = planar_scaling_on_path(poses=np.array([pose]), s=scale_factor)[0]
    #T, ax, scale_factor = get_pca_transform(pose[:,0], pose[:,1])
    #pose = planar_rigid_body_transformation_on_path(poses=np.array([pose]), Trans=T)[0]
    #pose = reform_starting_point(pose)
    #if ax is not None:
    #    ax.plot(pose[:,0], pose[:,1],'o',label=label + ' : Trans-Rotated')
    return pose

def reform_starting_point(pose):
    """ input: cupler curve trajectory of shape [num, 3]
        output: reformed, with changed starting point
    """
    leftmost = np.argmin(pose[:,0])
    if pose.shape[0] != 359:
        if (leftmost <= (pose.shape[0] - leftmost)):
            ax.plot(pose[leftmost,0], pose[leftmost,1], '*')
            ax.plot(pose[0,0], pose[0,1], '*')
            return pose
        else:
            ax.plot(pose[leftmost,0], pose[leftmost,1], '*')
            ax.plot(pose[0,0], pose[0,1], '*')
            return pose[::-1]
    else:
        new_pose = np.concatenate((pose[leftmost:],pose[:,leftmost]))
        new_pose = np.concatenate((new_pose,pose[leftmost]))

def truncate_curves(curv):
    """ curv is a list of len (num*3)
    """
    n  = len(curv)
    temp2 = []
    temp3 = []
    temp4 = []
    if n % 6 == 0:
        temp2.append(curv[:n/2])
        temp2.append(curv[n/2:])
    if n % 9 == 0:
        temp3.append(curv[:n/3])
        temp3.append(curv[n/3:2*n/3])
        temp3.append(curv[2*n/3:])
    if n % 12 == 0:
        temp4.append(curv[:n/4])
        temp4.append(curv[n/4:n/2])
        temp4.append(curv[n/2:3*n/4])
        temp4.append(curv[3*n/4:])
    return temp2, temp3, temp4

def transform_into_cylindrical_cord(poses):
    cylindrical_representation = np.zeros(poses.shape)
    cylindrical_representation[:,0] = poses[:,0]*np.cos(poses[:,2])
    cylindrical_representation[:,1] = poses[:,0]*np.sin(poses[:,2])
    cylindrical_representation[:,2] = poses[:,1]
    return cylindrical_representation

class CouplerCurves:
    def __init__(self):
        self.curv1 = []
        self.curv2 = []
        self.curv3 = []
        self.curv4 = []
        self.curv5 = []
        self.curv6 = []
        self.curv7 = []
        self.curv8 = []
        self.circuit = True
        self.signs = []
        self.crank_changed = False
    def push_point(self, points):
        if self.crank_changed:
            if self.circuit:
                self.curv5.append(points[0])
                self.curv6.append(points[1])
            else:
                self.curv7.append(points[0])
                self.curv8.append(points[1])
        else:
            if self.circuit:
                self.curv1.append(points[0])
                self.curv2.append(points[1])
            else:
                self.curv3.append(points[0])
                self.curv4.append(points[1])
    def change_crank(self):
        self.circuit = True
        self.crank_changed = True

    def change_circuit(self):
        if self.crank_changed:
            if len(self.curv5) != 0:
                self.circuit = not self.circuit
        else:
            if len(self.curv1) != 0:
                self.circuit = not self.circuit

    def finish(self):
        self.curv1 = np.array(self.curv1)
        self.curv2 = np.array(self.curv2)
        self.curv3 = np.array(self.curv3)
        self.curv4 = np.array(self.curv4)
        self.curv5 = np.array(self.curv5)
        self.curv6 = np.array(self.curv6)
        self.curv7 = np.array(self.curv7)
        self.curv8 = np.array(self.curv8)
        self.curves = [self.curv1, self.curv2, self.curv3, self.curv4, self.curv5, self.curv6, self.curv7, self.curv8]
        if len(self.curv1) >= 4:
            self.signs.append(signature(x=self.curv1[:,0], y=self.curv1[:,1], angle=self.curv1[:,2]))
            self.signs.append(signature(x=self.curv2[:,0], y=self.curv2[:,1], angle=self.curv2[:,2]))
        if len(self.curv3) >= 4:
            self.signs.append(signature(x=self.curv3[:,0], y=self.curv3[:,1], angle=self.curv3[:,2]))
            self.signs.append(signature(x=self.curv4[:,0], y=self.curv4[:,1], angle=self.curv4[:,2]))
        if len(self.curv5) >= 4:
            self.signs.append(signature(x=self.curv5[:,0], y=self.curv5[:,1], angle=self.curv5[:,2]))
            self.signs.append(signature(x=self.curv6[:,0], y=self.curv6[:,1], angle=self.curv6[:,2]))
        if len(self.curv7) >= 4:
            self.signs.append(signature(x=self.curv7[:,0], y=self.curv7[:,1], angle=self.curv7[:,2]))
            self.signs.append(signature(x=self.curv8[:,0], y=self.curv8[:,1], angle=self.curv8[:,2]))

    def plot_curves(self, ax, label='', mark='-*'):
        i = 0
        if len(self.curv1) != 0:
            ax.plot(self.curv1[:,0], self.curv1[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv2[:,0], self.curv2[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv3) != 0:
            ax.plot(self.curv3[:,0], self.curv3[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv4[:,0], self.curv4[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv5) != 0:
            ax.plot(self.curv5[:,0], self.curv5[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv6[:,0], self.curv6[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
        if len(self.curv7) != 0:
            ax.plot(self.curv7[:,0], self.curv7[:,1], mark , ms=1,lw=1, label=label+'%d'%i)
            i += 1
            ax.plot(self.curv8[:,0], self.curv8[:,1], mark , ms=1,lw=1, label=label+'%d'%i)

def simulate_fourbar(params, timing = None, both_branches=True):
    l1,l2,l3,l4,l5 = params
    coupler_curves = CouplerCurves()
    circuit_changed = False
    if timing is None:
        timing = np.arange(0.0, 2*np.pi, np.pi/180.0)
    for theta in timing:
        success, output = fourbar_fk(l1,l2,l3,l4,l5,theta)
        if success:
            coupler_curves.push_point(output)
            circuit_changed = False
        else:
            if not circuit_changed:
                coupler_curves.change_circuit()
                circuit_changed = True
    if both_branches:
        circuit_changed = False
        coupler_curves.change_crank()
        for theta in timing:
            success, output = fourbar_fk(l2,l1,l3,-l4,l5,theta)
            if success:
                coupler_curves.push_point(output)
                circuit_changed = False
            else:
                if not circuit_changed:
                    coupler_curves.change_circuit()
                    circuit_changed = True
    coupler_curves.finish()
    return coupler_curves

def fourbar_fk(l1,l2,l3,l4,l5,theta):
    """ Calculates coupler position and angle (floating link)
        returns a dict of coupler_points and coupler_angles
    """
    # fg = Crank ground joint
    # sg = Second ground joind
    fg = np.array([0, 0])
    sg = np.array([1, 0])
    # First Floating Point
    fe = np.array([l1*np.cos(theta), l1*np.sin(theta)])
    condition = (l2 + l3 > getDistance(sg,fe)) and (abs(l2 - l3) < getDistance(sg,fe))
    if not condition:
        return False, [0]

    x_inclination = np.arctan2(sg[1]-fe[1], sg[0]-fe[0])
    x1, y1 = fe
    x2, y2 = sg
    r, R = l2, l3
    d = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    x = (d**2-r**2+R**2)/(2.0*d)
    a = 1/d * np.sqrt(4*(d**2)*(R**2)-np.power((d**2)-(r**2)+(R**2),2))
    temp = getEndPoint(fe, x, x_inclination)
    # Second Floating Points
    se1 = getEndPoint(temp, a/2.0, x_inclination + np.pi/2.0)
    se2 = getEndPoint(temp, a/2.0, x_inclination - np.pi/2.0)

    l3_inclination1 = np.arctan2(se1[1]-fe[1], se1[0]-fe[0])
    l3_inclination2 = np.arctan2(se2[1]-fe[1], se2[0]-fe[0])
    temp = getEndPoint((fe+se1)/2, l4, l3_inclination1)
    c1 = getEndPoint(temp, l5, l3_inclination1 + np.pi/2.0)
    temp = getEndPoint((fe+se2)/2, l4, l3_inclination2)
    c2 = getEndPoint(temp, l5, l3_inclination2 + np.pi/2.0)
    return True, np.array([[c1[0], c1[1], l3_inclination1], [c2[0], c2[1], l3_inclination2]])

def getEndPoint(startPoint, length, angle):
    return np.array([startPoint[0] + (length * np.cos(angle)), startPoint[1] + (length * np.sin(angle))]);

def getDistance(pt1,pt2):
        return np.sqrt(np.sum(np.power(pt1-pt2,2)));

def getMotionError(x, *args):
    """
    Objective function of optimization routine which accepts
    target path signature and link params namely l1,l2,l3,l4,l5
    and returns an error function

    x : params
    target_path_sign : args[0]
    verbose: args[1]
    reversible: args[2]

    """
    target_path_sign = args[0]
    verbose = False
    reversible = False
    if len(args) > 1:
        verbose = args[1]
        reversible = args[2]
    min_score = 100
    coupler_curves = simulate_fourbar(x)
    i = 0
    index = 0
    for sign in coupler_curves.signs:
        if(len(sign['motion_sign'])*1.3 >= len(target_path_sign['motion_sign']) or reversible):
            score = motion_cross_corelation(sign, target_path_sign)['distance']
            if verbose:
                print('score for %d sign is %f:'%(i, score))
            if score < min_score:
                min_score = score
                index = i
        i += 1
    print('min_score : %f'%min_score)
    return min_score, index, coupler_curves

def getPathError(x, *args):
    """
    Objective function of optimization routine which accepts
    target path signature and link params namely l1,l2,l3,l4,l5
    and returns an error function

    x : params
    target_path_sign : args[0]
    verbose: args[1]
    reversible: args[2]

    """
    target_path_sign = args[0]
    verbose = False
    reversible = False
    if len(args) > 1:
        verbose = args[1]
        reversible = args[2]
    min_score = 3
    coupler_curves = simulate_fourbar(x)
    i = 0
    index = 0
    for sign in coupler_curves.signs:
        if(len(sign)*1.3 >= len(target_path_sign) or reversible):
            score = 1 - normalized_cross_corelation(sign, target_path_sign)['score']
            if verbose:
                print('score for %d sign is %f:'%(i, score))
            if score < min_score:
                min_score = score
                index = i
        i += 1
    #print('min_score : %f'%min_score)
    return min_score, index, coupler_curves

def getParams(linkage):
    link = json.loads(linkage)
    fe = np.array([link['linkageInfo'][1][3][1], link['linkageInfo'][1][4][1]])
    se = np.array([link['linkageInfo'][2][3][1], link['linkageInfo'][2][4][1]])
    l1, l2, l3 = np.sqrt(np.sum(fe**2)), np.sqrt(np.sum((se - [1,0])**2)), np.sqrt(np.sum((se - fe)**2))
    lc, ang = np.array([link['linkageInfo'][3][1][1], link['linkageInfo'][3][2][1]])
    l5 = lc*np.sin(ang)
    l4 = lc*np.cos(ang) - l3/2.0
    return [l1,l2,l3,l4,l5]

if __name__=='__main__':
    import plot
    from matplotlib.patches import Ellipse
    import random
    x = np.load('../../data/npy/x.npy')
    y = np.load('../../data/npy/y.npy')
    theta = np.load('../../data/npy/theta.npy')
    print(x.shape)
    #distance_mat = np.load('../../data/npy/distance_matrix.npy')
    #model = AgglomerativeClustering(n_clusters=10, linkage="average", affinity='precomputed')

    #model.fit(distance_mat)

    #cluster_index = 1
    #j = 0
    #cluster = []
    #for i in model.labels_:
    #    if cluster_index == i:
    #        cluster.append(j)
    #    j += 1
    #c1 = np.argsort(distance_mat[cluster_index-1,:])

    #print np.sort(distance_mat[cluster_index-1,:])[-10:]
    #print c1[-10:]
    #q_ind = [6720, 0]
    #q_ind = [183, 4416, 7644, 8123, 9695, 10807, 11012, 15119, 15735, 18055, 19457, 19666, 19685, 24411, 24495, 24738, 25563, 25837, 26189, 29273]
    #q_ind = [5829, 9439, 18152, 18706, 25678, 27816]
    with open('../../data/cluster_base.pkl', 'rb') as f:
        cluster = pickle.load(f)
    #z_features = np.load('../../data/auto-encoder/motion_dnn/input.npy')
    #cluster = KMeans(n_clusters=1000)
    #cluster.fit(z_features)
    cluster_index = 11
    j = 0
    clust = []
    for i in cluster.labels_:
        if cluster_index == i:
            clust.append(j)
        j += 1
    print(clust)
    q_ind = [clust[:100]]
    qx, qy, qtheta = x[q_ind].tolist(), y[q_ind].tolist(), theta[q_ind].tolist()
    fig, ax = plot.ax2d('Trajectories', axis='equal')
    fig2, ax2 = plot.ax2d('Signatures of Trajectory')
    fig3, ax3 = plot.ax2d('Orientation vs Curvature Integral')
    fig4, ax4 = plot.ax2d('Correlation')
    signs = []
    i = 0
    for xq, yq, thetaq in zip(qx, qy, qtheta):
        traj = transform_poses(xq, yq, thetaq, ax=ax, label='%d'%i)
        signs.append(signature(x=traj[:,0], y=traj[:,1], angle=traj[:,2], ax=ax2, ax2=ax, ax3=ax3, mark='-', label='%d'%i))
        i += 1
        #signs.append(signature(x=-traj[:,0], y=traj[:,1], angle=traj[:,2], ax2=ax, mark='-', label='%d'%i))
        #j = 18
        #traj = transform_poses(xq[:j]*2, yq[:j]*2, thetaq[:j], ax=ax, label='%d'%i)
        #sign_2 = signature(x=traj[:,0], y=traj[:,1], angle=traj[:,2], ax=ax2, ax2=ax, ax3=ax3, mark='-', label='%d'%i)
        #normalized_cross_corelation(sign_1, sign_2, ax=ax2, ax2=ax4)
        #i += 1
    similarity = normalized_cross_corelation(signs[0], signs[1], ax2=ax4)['score']
    #print similarity
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax.legend(loc='best')
    plot.plt.show()

