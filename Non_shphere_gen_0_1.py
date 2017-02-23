# -*- coding: utf-8 -*-
"""
Non-spherical particle generator
3D scanning stl &
Particle Shape Characterisation

@author: Dr Wei Chen
Centre for Bulk Solids & Particulate Technologies
Newcastle Institute for Energy and Resources
The University of Newcastle
Version 0.1
"""

import numpy as np
import trimesh
from scipy.spatial import Delaunay
import math
import time

# Stage 1: Sphere sampling
#---------------------------------------------------------------------------------------------------#
def collinear(points_idx, vertices):
    # if collinear, return 1
    # if not collinear return 0
    p1 = vertices[points_idx[0]]
    p2 = vertices[points_idx[1]]
    p3 = vertices[points_idx[2]]
    p4 = vertices[points_idx[3]]
    # check colinear
    p123 = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
    p124 = p1[0]*(p2[1]-p4[1]) + p2[0]*(p4[1]-p1[1]) + p4[0]*(p1[1]-p2[1])
    p134 = p1[0]*(p4[1]-p3[1]) + p4[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p4[1])
    p234 = p4[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p4[1]) + p3[0]*(p4[1]-p2[1])
    
    collinear = p123 and p124 and p134 and p234
    if collinear != 0:
        return(collinear)
    else:
        return(1)
#---------------------------------------------------------------------------------------------------#
def coplanar(points_idx, vertices):
    # 2. 4 points are not coplanar
    # get four vertices
    p1 = vertices[points_idx[0]]
    p2 = vertices[points_idx[1]]
    p3 = vertices[points_idx[2]]
    p4 = vertices[points_idx[3]]

#    vrtcs = np.array([p1, p2, p3, p4])
#    tri = Delaunay(vrtcs, qhull_options='Qbb')
#    cp = tri.coplanar
    tet_vol = np.array([np.hstack((p1,1)), np.hstack((p2,1)), np.hstack((p3,1)), np.hstack((p4,1))])
    
    Vol = np.linalg.det(tet_vol)
    if Vol > 0:
        return(0)
    else:
        return(1)
#---------------------------------------------------------------------------------------------------#
def sphere_coord(points_idx, vertices):
    # get four vertices
    p1 = vertices[points_idx[0]]
    p2 = vertices[points_idx[1]]
    p3 = vertices[points_idx[2]]
    p4 = vertices[points_idx[3]]
    # sphere coordinates calculated based on
    # http://www.ambrsoft.com/TrigoCalc/Sphere/Spher3D_.htm
    t1 = -(p1[0]**2 + p1[1]**2 + p1[2]**2)
    t2 = -(p2[0]**2 + p2[1]**2 + p2[2]**2)
    t3 = -(p3[0]**2 + p3[1]**2 + p3[2]**2)
    t4 = -(p4[0]**2 + p4[1]**2 + p4[2]**2)
    
    T = np.linalg.det(np.array([[p1[0], p1[1], p1[2], 1], [p2[0], p2[1], p2[2], 1], [p3[0], p3[1], p3[2], 1], [p4[0], p4[1], p4[2], 1]]))
    D = np.linalg.det(np.array([[t1, p1[1], p1[2], 1], [t2, p2[1], p2[2], 1], [t3, p3[1], p3[2], 1], [t4, p4[1], p4[2], 1]]))/T
    E = np.linalg.det(np.array([[p1[0], t1, p1[2], 1], [p2[0], t2, p2[2], 1], [p3[0], t3, p3[2], 1], [p4[0], t4, p4[2], 1]]))/T
    F = np.linalg.det(np.array([[p1[0], p1[1], t1, 1], [p2[0], p2[1], t2, 1], [p3[0], p3[1], t3, 1], [p4[0], p4[1], t4, 1]]))/T
    G = np.linalg.det(np.array([[p1[0], p1[1], p1[2], t1], [p2[0], p2[1], p2[2], t2], [p3[0], p3[1], p3[2], t3], [p4[0], p4[1], p4[2], t4]]))/T
    
    centre = [-D/2, -E/2, -F/2]
    diameter = abs(math.sqrt(pow(D,2) + pow(E,2) + pow(F,2) - 4*G))
    sphere = {'centre':centre, 'diameter':diameter}
    return(sphere)
#---------------------------------------------------------------------------------------------------#           



# Stage 2: Sphere culling
#---------------------------------------------------------------------------------------------------#
# Step 1
def cvnx_hull(sphere, cvnx_hull_faces, vtxs_coord, bounds):
    # get the sphere information
    centre = sphere['centre']
    diameter = sphere['diameter']
    # computer the distances
    Dch = []
    # loop through face vertices and faces to find minDch
    # get the convex_hull of the mesh
    #cvnx_hull = mesh.convex_hull
    #cvnx_hull_faces = cvnx_hull.faces
    #cvnx_hull_facesN = cvnx_hull.face_normals
    #fc = 0
    for fcN in cvnx_hull_faces:
        # get the three points defining this face
        point_1 = vtxs_coord[fcN[0]]
        point_2 = vtxs_coord[fcN[1]]
        point_3 = vtxs_coord[fcN[2]]
        # compute vector
        vector_1 = point_1 - point_2
        vector_2 = point_1 - point_3
        # compute normal vector
        n_vector = np.cross(vector_1, vector_2)
        A = n_vector[0]
        B = n_vector[1]
        C = n_vector[2]
        D = A*-point_1[0] + B*-point_1[1] + C*-point_1[2]
        dist = abs(A*centre[0]+B*centre[1]+C*centre[2]+D)/math.sqrt(A**2+B**2+C**2)
        #dist = np.linalg.norm(fcN - centre)
        Dch.append(dist - diameter/2)
    # Calc Tch
    Tch = min(abs(bounds[0] - bounds[1]))*0.3 # Key parameter to tune

    if np.amin(np.array(Dch)) > -Tch:
        return(1) # Keep this sphere
    else:
        return(0) # remve this sphere
    
#---------------------------------------------------------------------------------------------------#
# Step 2 Test sphere centre in convex_hull 
def P_in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    if hull.find_simplex(p)>=0:
        return(1)
    else:
        return(0)

#---------------------------------------------------------------------------------------------------#
def dist_vtx_sph(sphere, vertices, bounds):
    # get the sphere information
    centre = sphere['centre']
    # Calc sphere to vertex distance
    dist_v = vertices - centre
    dist = np.linalg.norm(dist_v, ord=None, axis=1)
    idx = 0
    vtx_remove = []
    # define the vertex removal threshold
    thresh = min(abs(bounds[0] - bounds[1]))*0.3 # Key parameter to tune
    # 0.05 defines as the 5% of the minimum bound limit
    for d in dist:
        if d < thresh:
            vtx_remove.append(idx)
        idx += 1
    return(vtx_remove)
    
    

#--------------------------------------------__main__-----------------------------------------------#
# specify the particle stl template
tic = time.clock()
# load the mesh
t_mesh = trimesh.load_mesh('test.stl')
# triangular faces on mesh
t_mesh_face = t_mesh.faces
t_mesh_face = t_mesh_face[0:10]
# triangular vertices on mesh
vertices = t_mesh.vertices
# bounds
bounds = t_mesh.bounds
# convex hull and convex hull faces
cvnx_hull_faces = t_mesh.convex_hull.faces
# numpy array to store the removed vertices
rmd_vtx = np.array([])
# numpy array to store the vertices defining the sphere
sp_vtxs_list = np.array([[0, 0, 0, 0]])
# numpy array to store the sphere centre and diameter
sphere_list  = np.array([0, 0, 0, 0])
# total vertice list
vert_list = range(0, len(vertices), 1)
# now get a loop to test distance between spheres and vertex
# Step 1 sphere generation 
# this step will check collinear and coplanar properties
# get all vertices from mesh
# find one set of points satisfying criteria
# randomly select 4 points
#### select list
for face in t_mesh_face:
    vert_list = np.delete(vert_list, [face[0], face[1], face[2]])
    print(face)
    for v in vert_list:
        
        #print(v)
        points_iter = np.concatenate((face, [v]))
        
        if np.logical_and(collinear(points_iter, vertices), coplanar(points_iter, vertices)):
            continue
        else:
            sphere = sphere_coord(points_iter, vertices)
            # Step sphere culling
            # Tch is the threshold of minimum distance between convex hull plane and the sphere
            # Tch needs to be defined
            convexh_test = cvnx_hull(sphere, cvnx_hull_faces, vertices, bounds)
            In_hull_test = P_in_hull(sphere['centre'], vertices)
            if np.logical_and(In_hull_test, convexh_test):
                vtx_remove = np.array(dist_vtx_sph(sphere, vertices, bounds))
                rmd_vtx = np.concatenate((rmd_vtx, vtx_remove))
                rmd_vtx = np.unique(rmd_vtx)
                print('Eleminated ' + str(len(rmd_vtx)) + ' out of ' + str(len(vertices)) + ' the total vertices')
                sphere_list_tmp = np.hstack((sphere['centre'],sphere['diameter']/2))
                sphere_list = np.vstack((sphere_list, sphere_list_tmp))
                print('Found ' + str(len(sphere_list)) + ' spheres')
                    
                    
# final sphere list
sphere_list = np.delete(sphere_list, 0, axis = 0)
np.savetxt("sphere.csv", sphere_list, delimiter=" ")
toc = time.clock()
print('Processing time is ' + str(toc - tic))