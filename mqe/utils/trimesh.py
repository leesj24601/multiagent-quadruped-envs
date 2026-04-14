""" This file defines a mesh as a tuple of (vertices, triangles)
All operations are based on numpy ndarray
- vertices: np ndarray of shape (n, 3) np.float32
- triangles: np ndarray of shape (n_, 3) np.uint32
"""
import numpy as np

def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]

    return vertices, triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(trimeshes[1:])
        )

    # only two trimesh to combine
    trimesh_0, trimesh_1 = trimeshes
    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    """ inplace operation """
    trimesh[0] += move

def wall_with_hole_trimesh(
        size, # [thickness(x), width(y), height(z)]
        hole_radius,
        hole_center_y, # relative to wall center
        hole_center_z, # relative to wall center
        center_position,
        resolution=32
    ):
    """
    Generates a trimesh for a wall with a circular hole using radial projection.
    The wall is aligned with y-z plane, thickness along x.
    """
    thickness, width, height = size
    cx, cy, cz = center_position
    
    # Vertices
    verts = []
    
    def add_vert(x, y, z):
        verts.append([x, y, z])
        return len(verts) - 1

    # We will create vertices for Front Face and Back Face.
    # For each face, we have 'resolution' segments.
    # Each segment has an inner point (on circle) and an outer point (on rectangle).
    
    # Pre-calculate relative coordinates for one face (2D y-z plane)
    # Center of hole is (hole_center_y, hole_center_z) relative to wall center.
    # But we want to project rays from the HOLE CENTER.
    
    # Outer rectangle bounds relative to wall center:
    # y: [-width/2, width/2]
    # z: [-height/2, height/2]
    
    # Bounds relative to HOLE CENTER:
    # y_min = -width/2 - hole_center_y
    # y_max = width/2 - hole_center_y
    # z_min = -height/2 - hole_center_z
    # z_max = height/2 - hole_center_z
    
    y_min = -width/2 - hole_center_y
    y_max = width/2 - hole_center_y
    z_min = -height/2 - hole_center_z
    z_max = height/2 - hole_center_z
    
    inner_indices_front = []
    outer_indices_front = []
    inner_indices_back = []
    outer_indices_back = []
    
    angle_step = 2 * np.pi / resolution
    
    for i in range(resolution):
        angle = i * angle_step
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Inner point (Circle)
        dy_in = hole_radius * cos_a
        dz_in = hole_radius * sin_a
        
        # Outer point (Rectangle Projection)
        # Ray is (cos_a, sin_a)
        # We want t such that (t*cos_a, t*sin_a) hits the box [y_min, y_max] x [z_min, z_max]
        
        # Check intersection with vertical lines y = y_max or y = y_min
        t_y = float('inf')
        if cos_a > 1e-6: # Moving Right
            t_y = y_max / cos_a
        elif cos_a < -1e-6: # Moving Left
            t_y = y_min / cos_a
            
        # Check intersection with horizontal lines z = z_max or z = z_min
        t_z = float('inf')
        if sin_a > 1e-6: # Moving Up
            t_z = z_max / sin_a
        elif sin_a < -1e-6: # Moving Down
            t_z = z_min / sin_a
            
        t = min(t_y, t_z)
        
        dy_out = t * cos_a
        dz_out = t * sin_a
        
        # Absolute coordinates (relative to wall center)
        y_in = hole_center_y + dy_in
        z_in = hole_center_z + dz_in
        y_out = hole_center_y + dy_out
        z_out = hole_center_z + dz_out
        
        # Add vertices
        # Front (+thickness/2)
        inner_indices_front.append(add_vert(thickness/2, y_in, z_in))
        outer_indices_front.append(add_vert(thickness/2, y_out, z_out))
        
        # Back (-thickness/2)
        inner_indices_back.append(add_vert(-thickness/2, y_in, z_in))
        outer_indices_back.append(add_vert(-thickness/2, y_out, z_out))
        
    triangles = []
    
    # Helper for quad (CCW)
    def add_quad(v1, v2, v3, v4):
        triangles.append([v1, v2, v3])
        triangles.append([v1, v3, v4])
        
    for i in range(resolution):
        next_i = (i + 1) % resolution
        
        # 1. Front Face
        # Outer ring -> Inner ring
        # Normal +x
        # Vertices: out[i], out[next], in[next], in[i]
        add_quad(outer_indices_front[i], outer_indices_front[next_i], 
                 inner_indices_front[next_i], inner_indices_front[i])
                 
        # 2. Back Face
        # Normal -x
        # Vertices: out[i], in[i], in[next], out[next] (reversed winding)
        add_quad(outer_indices_back[i], inner_indices_back[i],
                 inner_indices_back[next_i], outer_indices_back[next_i])
                 
        # 3. Inner Cylinder (Hole)
        # Connect inner_front to inner_back
        # Normal pointing INWARD (towards center)
        # Vertices: in_f[i], in_f[next], in_b[next], in_b[i]
        # Check winding:
        # v1 = in_f[next] - in_f[i] (tangent)
        # v2 = in_b[i] - in_f[i] (-x)
        # v1 x v2 -> roughly radial OUT?
        # Let's try: in_f[next], in_f[i], in_b[i]
        # Same as previous logic.
        add_quad(inner_indices_front[next_i], inner_indices_front[i],
                 inner_indices_back[i], inner_indices_back[next_i])
                 
        # 4. Outer Box Sides
        # Connect outer_front to outer_back
        # Normal pointing OUTWARD
        # Vertices: out_f[i], out_b[i], out_b[next], out_f[next]
        add_quad(outer_indices_front[i], outer_indices_back[i],
                 outer_indices_back[next_i], outer_indices_front[next_i])

    # Convert to numpy
    vertices = np.array(verts, dtype=np.float32)
    # Add center position
    vertices += np.array(center_position, dtype=np.float32)
    
    triangles = np.array(triangles, dtype=np.uint32)
    
    return vertices, triangles
