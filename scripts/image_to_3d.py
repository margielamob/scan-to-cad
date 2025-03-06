#!/usr/bin/env python3
# image_to_3d.py - Script for converting images to 3D models

import sys
import os
import json
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2

# Check if OpenCascade is available
try:
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static_SetCVal
    from OCC.Core.IFSelect import IFSelect_RetDone
    has_occ = True
except ImportError:
    has_occ = False
    print("Warning: OpenCascade binding not found. Will use STL export instead of STEP.")

import sys
import os
import json
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_image(image_path):
    """Load an image and convert to grayscale"""
    log(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log(f"Loaded image with shape: {img.shape}")
    return img, gray

def generate_depth_map(gray_img, method="sobel"):
    """Generate a depth map from a grayscale image using edge detection"""
    log(f"Generating depth map using {method} method")
    
    if method == "sobel":
        # Apply Sobel operator for edge detection
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine to get gradient magnitude
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Invert so edges are high (far from camera)
        depth_map = 255 - gradient
        
    elif method == "canny":
        # Apply Canny edge detection
        edges = cv2.Canny(gray_img, 50, 150)
        
        # Invert to get depth map (edges are high)
        depth_map = 255 - edges
        
        # Apply Gaussian blur to smooth
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
    else:
        # Simple intensity as depth (brighter = closer)
        depth_map = gray_img
    
    # Apply distance transform to create smoother depth gradients
    # Threshold first to create binary image of edges
    _, thresh = cv2.threshold(255 - depth_map, 30, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    
    # Normalize and invert so edges are higher
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    depth_map = 1.0 - dist_transform
    
    return depth_map

def create_point_cloud_from_depth(img, depth_map, downsample_factor=4):
    """Create a point cloud from an image and its depth map"""
    log("Creating point cloud from depth map")
    
    # Downsample for performance
    height, width = depth_map.shape
    new_height, new_width = height // downsample_factor, width // downsample_factor
    img_small = cv2.resize(img, (new_width, new_height))
    depth_small = cv2.resize(depth_map, (new_width, new_height))
    
    # Create arrays for points and colors
    points = []
    colors = []
    
    # Scale factor for depth
    depth_scale = 0.3
    
    for y in range(new_height):
        for x in range(new_width):
            # Calculate 3D point from pixel coordinates and depth
            depth_value = depth_small[y, x] * depth_scale
            points.append([x / new_width - 0.5, y / new_height - 0.5, depth_value])
            
            # Get color from original image
            color = img_small[y, x] / 255.0
            colors.append(color)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    log(f"Created point cloud with {len(pcd.points)} points")
    return pcd

def create_mesh_from_point_cloud(pcd):
    """Create a mesh from a point cloud using Poisson surface reconstruction"""
    log("Creating mesh from point cloud")
    
    # Estimate normals if they don't exist
    if not pcd.has_normals():
        log("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
    
    # Poisson surface reconstruction
    log("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    
    # Remove low density vertices
    log("Removing low density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean up the mesh
    mesh.compute_vertex_normals()
    
    log(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def image_to_3d(input_file, output_file, params=None):
    """Main function to convert an image to 3D model"""
    start_time = datetime.now()
    log(f"Starting conversion of {input_file} to {output_file}")
    
    # Set default parameters
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    depth_method = params.get('depth_method', 'sobel')
    downsample_factor = params.get('downsample_factor', 4)
    
    try:
        # Load and process the image
        img, gray = load_image(input_file)
        
        # Generate depth map
        depth_map = generate_depth_map(gray, method=depth_method)
        
        # Create point cloud from depth map
        pcd = create_point_cloud_from_depth(img, depth_map, downsample_factor)
        
        # Create mesh from point cloud
        mesh = create_mesh_from_point_cloud(pcd)
        
        # Export the mesh to the requested format
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
            'stats': {
                'points': len(pcd.points),
                'vertices': len(mesh.vertices),
                'triangles': len(mesh.triangles)
            }
        }
        
        return result
        
    except Exception as e:
        log(f"Error in image_to_3d: {str(e)}")
        return {
            'input_file': input_file,
            'output_file': output_file,
            'duration': (datetime.now() - start_time).total_seconds(),
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python image_to_3d.py <input_file> <output_file> [params_json]")
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
    result = image_to_3d(input_file, output_file, params)
    
    # Output result as JSON for the Node.js server to parse
    print(json.dumps(result))
    
    # Exit with appropriate code
    sys.exit(0 if result and result.get('success', False) else 1)

def export_mesh(mesh, output_file):
    """Export the mesh to a file"""
    log(f"Exporting mesh to: {output_file}")
    
    # If output is requested as STEP but OCC is not available, use STL instead
    if output_file.lower().endswith('.step') and not has_occ:
        output_file = output_file.rsplit('.', 1)[0] + '.stl'
        log(f"OpenCascade not available, exporting to STL instead: {output_file}")
    
    # Export based on file extension
    try:
        # For STEP files, we need special handling
        if output_file.lower().endswith('.step') and has_occ:
            # For now, we just export to STL since proper STEP conversion would
            # require more complex feature detection
            tmp_stl = output_file.rsplit('.', 1)[0] + '.stl'
            o3d.io.write_triangle_mesh(tmp_stl, mesh)
            log(f"Exported STL mesh for STEP conversion: {tmp_stl}")
            
            # TODO: Implement proper STEP conversion with OpenCascade
            # For now, just copy the STL to the output file as placeholder
            with open(tmp_stl, 'rb') as src, open(output_file, 'wb') as dst:
                dst.write(src.read())
                
            log(f"Note: Proper STEP conversion not implemented. Created placeholder file.")
            return True
            
        else:
            # Standard mesh export
            o3d.io.write_triangle_mesh(output_file, mesh)
            log(f"Exported mesh as: {output_file}")
            return True
            
    except Exception as e:
        log(f"Error exporting mesh: {str(e)}")
        return False