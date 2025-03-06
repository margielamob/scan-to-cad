#!/usr/bin/env python3
# scan_to_cad.py - Main conversion script for Scan-to-CAD processing

import sys
import os
import json
import numpy as np
import open3d as o3d
from datetime import datetime

# Import OCC for CAD operations
try:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2, gp_Circ
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static_SetCVal
    from OCC.Core.IFSelect import IFSelect_RetDone
    has_occ = True
except ImportError:
    has_occ = False
    print("Warning: OpenCascade binding not found. Will use STL export instead of STEP.")

# Define the logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def preprocess_mesh(mesh, voxel_size=0.01):
    """Preprocess the mesh to clean and prepare for feature detection"""
    log("Preprocessing mesh...")
    
    # Remove duplicated vertices and triangles
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    
    # Ensure normals are computed
    mesh.compute_vertex_normals()
    
    # Compute a point cloud from the mesh for some operations
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    
    # Voxel downsampling for efficiency
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Remove statistical outliers
    pcd, indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    log(f"Preprocessed point cloud has {len(pcd.points)} points")
    return pcd, mesh

def estimate_normals(pcd, radius=0.1, max_nn=30):
    """Estimate normals if not already present"""
    if not pcd.has_normals():
        log("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
    return pcd

def segment_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000, min_points=100):
    """Segment planes from the point cloud"""
    log("Segmenting planes...")
    planes = []
    remaining_cloud = pcd
    
    while len(remaining_cloud.points) > min_points:
        # Run RANSAC to find the largest plane
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) < min_points:
            break
            
        # Extract the planar inliers
        inlier_cloud = remaining_cloud.select_by_index(inliers)
        outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        
        # Get the plane equation: ax + by + cz + d = 0
        a, b, c, d = plane_model
        
        # Calculate the centroid of the plane segment
        points = np.asarray(inlier_cloud.points)
        centroid = np.mean(points, axis=0)
        
        # Calculate the area of the plane segment (approximate)
        hull, _ = inlier_cloud.compute_convex_hull(joggle_inputs=True)
        area = hull.get_surface_area()
        
        # Store the plane information
        planes.append({
            'type': 'plane',
            'equation': [a, b, c, d],
            'centroid': centroid.tolist(),
            'normal': [a, b, c],
            'points': len(inlier_cloud.points),
            'area': area,
            'inlier_cloud': inlier_cloud  # Keep reference to the point cloud
        })
        
        log(f"Found plane with {len(inliers)} points and area {area:.4f}")
        
        # Update the remaining cloud
        remaining_cloud = outlier_cloud
        
        # Stop if we've processed more than 90% of the original cloud
        if len(remaining_cloud.points) < 0.1 * len(pcd.points):
            break
        
    log(f"Segmented {len(planes)} planes")
    return planes, remaining_cloud

def detect_cylinders(pcd, min_radius=0.05, max_radius=1.0, distance_threshold=0.01, 
                    ransac_n=3, num_iterations=1000, min_points=100):
    """Detect cylinders in the point cloud"""
    log("Detecting cylinders...")
    cylinders = []
    remaining_cloud = pcd
    
    while len(remaining_cloud.points) > min_points:
        try:
            # Use RANSAC to detect a cylinder
            cylinder_model, inliers = remaining_cloud.segment_cylinder(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                min_radius=min_radius,
                max_radius=max_radius
            )
            
            if len(inliers) < min_points:
                break
                
            # Extract the cylinder inliers
            inlier_cloud = remaining_cloud.select_by_index(inliers)
            outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
            
            # Get cylinder parameters
            axis, center, radius = cylinder_model
            
            # Estimate the cylinder height by projecting points onto the axis
            points = np.asarray(inlier_cloud.points)
            center = np.array(center)
            axis = np.array(axis)
            
            # Project points onto the axis
            vectors = points - center
            projections = np.dot(vectors, axis)
            
            # Find min and max to get the height
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            height = max_proj - min_proj
            
            # Calculate the start and end points of the cylinder axis
            start_point = center + min_proj * axis
            end_point = center + max_proj * axis
            
            # Store the cylinder information
            cylinders.append({
                'type': 'cylinder',
                'axis': axis.tolist(),
                'center': center.tolist(),
                'radius': radius,
                'height': height,
                'start_point': start_point.tolist(),
                'end_point': end_point.tolist(),
                'points': len(inlier_cloud.points),
                'inlier_cloud': inlier_cloud  # Keep reference to the point cloud
            })
            
            log(f"Found cylinder with radius {radius:.4f}, height {height:.4f}, and {len(inliers)} points")
            
            # Update the remaining cloud
            remaining_cloud = outlier_cloud
            
        except Exception as e:
            log(f"Error in cylinder detection: {str(e)}")
            break
            
    log(f"Detected {len(cylinders)} cylinders")
    return cylinders, remaining_cloud

def detect_spheres(pcd, min_radius=0.05, max_radius=1.0, distance_threshold=0.01, 
                  ransac_n=4, num_iterations=1000, min_points=100):
    """Detect spheres in the point cloud"""
    log("Detecting spheres...")
    spheres = []
    remaining_cloud = pcd
    
    while len(remaining_cloud.points) > min_points:
        try:
            # Use RANSAC to detect a sphere
            sphere_model, inliers = remaining_cloud.segment_sphere(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                min_radius=min_radius,
                max_radius=max_radius
            )
            
            if len(inliers) < min_points:
                break
                
            # Extract the sphere inliers
            inlier_cloud = remaining_cloud.select_by_index(inliers)
            outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
            
            # Get sphere parameters (center, radius)
            center, radius = sphere_model
            
            # Store the sphere information
            spheres.append({
                'type': 'sphere',
                'center': center.tolist(),
                'radius': radius,
                'points': len(inlier_cloud.points),
                'inlier_cloud': inlier_cloud  # Keep reference to the point cloud
            })
            
            log(f"Found sphere with radius {radius:.4f} and {len(inliers)} points")
            
            # Update the remaining cloud
            remaining_cloud = outlier_cloud
            
        except Exception as e:
            log(f"Error in sphere detection: {str(e)}")
            break
            
    log(f"Detected {len(spheres)} spheres")
    return spheres, remaining_cloud

def create_step_file(features, output_file):
    """Create a STEP file from the detected features"""
    if not has_occ:
        log("OpenCascade not available. Skipping STEP creation.")
        return False
        
    log("Creating STEP file from detected features...")
    
    # Initialize the STEP writer
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")
    
    # Create shapes for each feature
    shapes = []
    
    # Process planes
    for i, plane in enumerate(features.get('planes', [])):
        try:
            # Extract plane parameters
            a, b, c, d = plane['equation']
            centroid = plane['centroid']
            
            # Create a normal vector for the plane
            normal = gp_Dir(a, b, c)
            
            # Create a point on the plane
            point = gp_Pnt(centroid[0], centroid[1], centroid[2])
            
            # Determine plane size (simplified approach)
            size = np.sqrt(plane['area'])
            
            # Create an orthogonal coordinate system
            z_axis = normal
            x_axis = gp_Dir(1, 0, 0)
            if abs(normal.Dot(x_axis)) > 0.9:
                x_axis = gp_Dir(0, 1, 0)
            y_axis = z_axis.Crossed(x_axis)
            x_axis = y_axis.Crossed(z_axis)
            
            # Create points for a rectangular face
            p1 = point.Translated(gp_Vec(-size/2, -size/2, 0))
            p2 = point.Translated(gp_Vec(size/2, -size/2, 0))
            p3 = point.Translated(gp_Vec(size/2, size/2, 0))
            p4 = point.Translated(gp_Vec(-size/2, size/2, 0))
            
            # Create edges
            edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
            edge3 = BRepBuilderAPI_MakeEdge(p3, p4).Edge()
            edge4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
            
            # Create wire
            wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3, edge4).Wire()
            
            # Create face
            face = BRepBuilderAPI_MakeFace(wire).Face()
            shapes.append(face)
            
            log(f"Created plane {i+1}")
            
        except Exception as e:
            log(f"Error creating plane {i+1}: {str(e)}")
    
    # Process cylinders
    for i, cylinder in enumerate(features.get('cylinders', [])):
        try:
            # Extract cylinder parameters
            center = cylinder['center']
            axis = cylinder['axis']
            radius = cylinder['radius']
            height = cylinder['height']
            start_point = cylinder['start_point']
            
            # Create a direction for the cylinder axis
            direction = gp_Dir(axis[0], axis[1], axis[2])
            
            # Create a point for the base of the cylinder
            base_point = gp_Pnt(start_point[0], start_point[1], start_point[2])
            
            # Create a coordinate system
            ax2 = gp_Ax2(base_point, direction)
            
            # Create the cylinder
            cylinder_shape = BRepPrimAPI_MakeCylinder(ax2, radius, height).Shape()
            shapes.append(cylinder_shape)
            
            log(f"Created cylinder {i+1}")
            
        except Exception as e:
            log(f"Error creating cylinder {i+1}: {str(e)}")
    
    # Process spheres
    for i, sphere in enumerate(features.get('spheres', [])):
        try:
            # Extract sphere parameters
            center = sphere['center']
            radius = sphere['radius']
            
            # Create a point for the center of the sphere
            center_point = gp_Pnt(center[0], center[1], center[2])
            
            # Create the sphere
            sphere_shape = BRepPrimAPI_MakeSphere(center_point, radius).Shape()
            shapes.append(sphere_shape)
            
            log(f"Created sphere {i+1}")
            
        except Exception as e:
            log(f"Error creating sphere {i+1}: {str(e)}")
    
    # Combine all shapes into a single solid
    if not shapes:
        log("No shapes to export")
        return False
        
    try:
        # Add the first shape
        result = shapes[0]
        
        # Fuse with remaining shapes
        for shape in shapes[1:]:
            result = BRepAlgoAPI_Fuse(result, shape).Shape()
        
        # Add to STEP writer
        step_writer.Transfer(result, STEPControl_AsIs)
        
        # Write the STEP file
        status = step_writer.Write(output_file)
        
        if status == IFSelect_RetDone:
            log(f"Successfully created STEP file: {output_file}")
            return True
        else:
            log(f"Error writing STEP file, status code: {status}")
            return False
            
    except Exception as e:
        log(f"Error creating STEP file: {str(e)}")
        return False

def export_mesh(mesh, output_file):
    """Export the mesh when OpenCascade is not available"""
    try:
        # If output is requested as STEP but OCC is not available, use STL instead
        if output_file.lower().endswith('.step') and not has_occ:
            output_file = output_file.rsplit('.', 1)[0] + '.stl'
        
        # Export the mesh
        o3d.io.write_triangle_mesh(output_file, mesh)
        log(f"Exported mesh to {output_file}")
        return True
    except Exception as e:
        log(f"Error exporting mesh: {str(e)}")
        return False

def convert_scan_to_cad(input_file, output_file, params=None):
    """Main function to convert a scan file to CAD format"""
    start_time = datetime.now()
    log(f"Starting conversion of {input_file} to {output_file}")
    
    # Set default parameters
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    voxel_size = params.get('voxel_size', 0.01)
    plane_distance = params.get('plane_distance', 0.02)
    cylinder_distance = params.get('cylinder_distance', 0.01)
    min_points = params.get('min_points', 100)
    
    # Load the input file
    log(f"Loading input file: {input_file}")
    ext = os.path.splitext(input_file)[1].lower()
    
    try:
        if ext in ['.ply', '.stl', '.obj']:
            # Load as mesh
            mesh = o3d.io.read_triangle_mesh(input_file)
            if not mesh.has_triangles():
                log("Error: Loaded mesh has no triangles")
                return False
            
            log(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
            
        elif ext in ['.pcd', '.xyz', '.pts']:
            # Load as point cloud
            pcd = o3d.io.read_point_cloud(input_file)
            if len(pcd.points) == 0:
                log("Error: Loaded point cloud is empty")
                return False
                
            log(f"Loaded point cloud with {len(pcd.points)} points")
            
            # Create a mesh from the point cloud using Poisson surface reconstruction
            log("Reconstructing mesh from point cloud...")
            pcd = estimate_normals(pcd)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            log(f"Reconstructed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
            
        else:
            log(f"Unsupported file format: {ext}")
            return False
            
    except Exception as e:
        log(f"Error loading file: {str(e)}")
        return False
    
    # Preprocess the mesh and convert to point cloud for feature detection
    pcd, mesh = preprocess_mesh(mesh, voxel_size)
    
    # Detect features
    features = {}
    
    # Detect planes
    planes, remaining_cloud = segment_planes(pcd, distance_threshold=plane_distance, min_points=min_points)
    features['planes'] = planes
    
    # Detect cylinders from the remaining points
    cylinders, remaining_cloud = detect_cylinders(remaining_cloud, distance_threshold=cylinder_distance, min_points=min_points)
    features['cylinders'] = cylinders
    
    # Detect spheres from the remaining points
    spheres, remaining_cloud = detect_spheres(remaining_cloud, distance_threshold=cylinder_distance, min_points=min_points)
    features['spheres'] = spheres
    
    # Create a feature report
    feature_report = {
        'planes': len(planes),
        'cylinders': len(cylinders),
        'spheres': len(spheres),
        'remaining_points': len(remaining_cloud.points)
    }
    log(f"Feature detection summary: {json.dumps(feature_report)}")
    
    # Export to the requested format
    success = False
    if output_file.lower().endswith('.step') and has_occ:
        success = create_step_file(features, output_file)
    else:
        # If STEP export failed or is not available, export the mesh
        success = export_mesh(mesh, output_file)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    log(f"Conversion completed in {duration:.2f} seconds. Success: {success}")
    
    # Create a result summary
    result = {
        'input_file': input_file,
        'output_file': output_file,
        'duration': duration,
        'success': success,
        'features': feature_report
    }
    
    return result

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python scan_to_cad.py <input_file> <output_file> [params_json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse optional parameters
    params = {}
    if len(sys.argv) > 3:
        try:
            params = json.loads(sys.argv[3])
        except:
            print("Error parsing parameters JSON")
    
    # Run the conversion
    result = convert_scan_to_cad(input_file, output_file, params)
    
    # Output result as JSON for the Node.js server to parse
    print(json.dumps(result))
    
    # Exit with appropriate code
    sys.exit(0 if result and result.get('success', False) else 1)