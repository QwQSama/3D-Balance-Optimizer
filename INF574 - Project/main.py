import numpy as np
import open3d as o3d
import os
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from voxelization_and_filling import voxel_carving


#transparent material
mat_tran = o3d.visualization.rendering.MaterialRecord()
mat_tran.shader = 'defaultLitTransparency'
mat_tran.base_color = [0.467, 0.467, 0.467, 0.2]
mat_tran.base_roughness = 0.0
mat_tran.base_reflectance = 0.0
mat_tran.base_clearcoat = 1.0
mat_tran.thickness = 1.0
mat_tran.transmission = 1.0
mat_tran.absorption_distance = 10
mat_tran.absorption_color = [0.5, 0.5, 0.5]
#defualt material
mat_default = o3d.visualization.rendering.MaterialRecord()

#input file name
input_filename = "spheres"

#define input & output file paths
input_path = "./Input/" + input_filename + ".obj"
input_mesh = o3d.io.read_triangle_mesh(input_path)
output_path_solution = os.path.abspath("./Output/"+ input_filename +"_solution.stl")

#define a plane
input_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#calculate euler angles
plane_normal_default = np.cross(input_array[0] - input_array[1], input_array[2] - input_array[1])
zy_projection = plane_normal_default - np.dot(np.dot(plane_normal_default, [1, 0, 0]), [1, 0, 0])
x_rotate_angle = np.dot(zy_projection, [0, 0, 1]) / (np.linalg.norm(zy_projection) * np.linalg.norm([0, 0, 1]))
zx_projection = plane_normal_default - np.dot(np.dot(plane_normal_default, [0, 1, 0]), [0, 1, 0])
y_rotate_angle = np.dot(zx_projection, [0, 0, 1]) / (np.linalg.norm(zx_projection) * np.linalg.norm([0, 0, 1]))

#rotate
rotate_matrix = input_mesh.get_rotation_matrix_from_xyz((x_rotate_angle, y_rotate_angle, 0))
input_mesh.rotate(rotate_matrix)

#translate
A = plane_normal_default[0]
B = plane_normal_default[1]
C = plane_normal_default[2]
D = -(A * input_array[1][0] + B * input_array[1][1] + C * input_array[1][2])

input_center = input_mesh.get_center()
original_distance = abs(A * input_center[0] + B * input_center[1] + C * input_center[2] + D) / np.sqrt(A * A + B * B + C * C)
new_distance = input_center[1]
translation_distance = original_distance - new_distance
input_mesh.translate((0, translation_distance, 0))

#test input
print(input_mesh)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
plane = o3d.geometry.TriangleMesh.create_box(width = 10, height = 0.01, depth = 10)
plane_translate = plane.translate((-5, 0, -5))
plane_translate.paint_uniform_color([0.1, 0.1, 0.7])
o3d.visualization.draw(
    [{'name': 'mesh', 'geometry': input_mesh, 'material': mat_tran}, 
    {'name': 'plane', 'geometry': plane, 'material': mat_default},
    {'name': 'frame', 'geometry': mesh_frame, 'material': mat_default}])

#voxelization & filling
visualization = True
cubic_size = 1.0
voxel_resolution = 128.0

voxel_grid, voxel_carving, voxel_fill = voxel_carving(
    input_mesh, cubic_size, voxel_resolution)

print("surface voxels")
print(voxel_fill)
o3d.visualization.draw_geometries([voxel_fill])



size = int(voxel_resolution*1.5)
voxels = voxel_fill.get_voxels()
vx_numpy = np.zeros((size,size,size))
for i in voxels: 
    index = i.grid_index 
    # print(voxel_grid.get_voxel_center_coordinate(index))
    if voxel_grid.get_voxel_center_coordinate(index)[2] >0:
        vx_numpy[index[0],index[1],index[2]] = 1
# print(vx_numpy)
print(np.sum(vx_numpy))

def voxel_surface_numpy(voxels_np):
    grid_shape = voxels_np.shape
    voxel_int = voxels_np.copy()
    voxel_surface = np.zeros_like(voxels_np)
    sum_nbh = lambda x,y,z,A: A[x+1,y,z] +A[x-1,y,z] +A[x,y+1,z] +A[x,y-1,z] + A[x,y,z+1] + A[x,y,z-1] 
    for id_x in range(grid_shape[0]-1):
        for id_y in range(grid_shape[1]-1):
            for id_z in range(grid_shape[2]-1):
                if(voxels_np[id_x,id_y,id_z]==1):
                    sum_neighboring_values = sum_nbh(id_x,id_y,id_z, voxels_np)
                    if(sum_neighboring_values<6):
                        voxel_int[id_x,id_y,id_z] = 0
                        voxel_surface[id_x,id_y,id_z] = 1

    return voxel_int, voxel_surface

voxel_int, voxel_surface = voxel_surface_numpy(vx_numpy);

# print(voxel_int)
# print(voxel_surface)
print(np.sum(voxel_int))
print(np.sum(voxel_surface))

def center_of_mass(voxels_np):
    grid_shape = voxels_np.shape
    counter = 0
    center = np.array([0.,0.,0.])
    for id_x in range(grid_shape[0]):
        for id_y in range(grid_shape[1]):
            for id_z in range(grid_shape[2]):
                #check whether we hit a a voxel in the mesh
                if(voxels_np[id_x,id_y,id_z]==1):
                    counter+=1
                    center = (1/float(counter))*np.array([float(id_x),float(id_y),float(id_z)]) + (float(counter-1)/float(counter))*center

    return([center[0],center[1],center[2]]) 

center = center_of_mass(vx_numpy)
print(center)

def polygon(voxels_np):
    size = voxels_np.shape[0]
    minz = size-1;
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if voxels_np[i,j,k] == 1:
                    if k < minz:
                        minz = k
    points = []
    for i in range(size):
        for j in range(size):
            if voxels_np[i,j,minz] == 1:
                points.append([i,j])
    
    hull = ConvexHull(points)
    points = np.array(points)
    res = []
    # plt.plot(points[:,0], points[:,1], 'o')
    res = points[hull.vertices]
    return res

support = polygon(voxel_surface)
print(support)

def center_of_polygon(points):
    n = len(points)
    counter = 0
    center = np.array([0.,0.])
    for i in range(n):
        counter+=1
        center = (1/float(counter))*np.array(points[i]) + (float(counter-1)/float(counter))*center

    return([center[0],center[1]]) 

print(center_of_polygon(support))

def cut_half(center_poly,voxel_int,voxels_np):
    grid_shape = voxels_np.shape
    counter = 0
    center = np.array([0.,0.,0.])
    for id_x in range(grid_shape[0]):
        for id_y in range(grid_shape[1]):
            for id_z in range(grid_shape[2]):
                #check whether we hit a a voxel in the mesh
                if(voxels_np[id_x,id_y,id_z]==1):
                    counter+=1
                    center = (1/float(counter))*np.array([float(id_x),float(id_y),float(id_z)]) + (float(counter-1)/float(counter))*center

    int_shape = voxel_int.shape
    distance = np.linalg.norm(center_poly-center[:2])

    for id_x in range(int_shape[0]):
        for id_y in range(int_shape[1]):
            for id_z in range(int_shape[2]):
                #check whether we hit a a voxel in the mesh
                if(voxel_int[id_x,id_y,id_z]==1):
                    counter_new = counter-1
                    center_new = center.copy()
                    center_new = -(1/float(counter_new))*np.array([float(id_x),float(id_y),float(id_z)]) + (float(counter_new+1)/float(counter_new))*center_new
                    distance_new = np.linalg.norm(center_new[:2]-center_poly)
                    if distance_new < distance:
                        voxel_int[id_x,id_y,id_z] = 0
                        counter = counter_new
                        center = center_new
                        distance = distance_new
    
    return voxel_int

res = cut_half(center_of_polygon(support),voxel_int,vx_numpy)
print(np.sum(res))