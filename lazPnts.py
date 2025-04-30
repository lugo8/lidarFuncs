import laspy
import numpy as np
import open3d as o3d
import math
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

class lazPnts:
    def __init__(self, input):
        #input a string detailing file name or a np.ndarray to make the points in lazPnts class

        if isinstance(input, str):
            file = input

            # Select the proper backend for laz files
            backend = laspy.LazBackend.Lazrs  # Or laspy.LazBackend.Laszip if using laszip

            # Open the .laz file with explicit backend selection
            try:
                with laspy.open(file, mode="r", laz_backend=backend) as file:
                    las = file.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                exit()

            # Extract point data (X, Y, Z coordinates)
            self.points = np.vstack((las.x, las.y, las.z)).transpose()

        elif isinstance(input, np.ndarray):
            #Allow input to be a numpy array of 3d points of format:
            # [[x1,y1,z1], [x2,y2,z2], ...]
            self.points = input

    def getPnts(self):
        #Returns the numpy array of 3d points
        return self.points

    def visualize(self):
        #Show the pointcloud
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Apply Gaussian denoising using a statistical outlier removal filter
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Optional: Recompute normals to smooth the surface
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Optionally, you can orient the normals (this ensures they point outward)
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))


        # # Normalize intensity values for visualization
        # intensity = np.array(las.intensity)
        # intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        # colors = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=-1)
        x_normalized = (self.points[:, 0] - np.min(self.points[:, 0])) / (np.max(self.points[:, 0]) - np.min(self.points[:, 0]))
        y_normalized = (self.points[:, 1] - np.min(self.points[:, 1])) / (np.max(self.points[:, 1]) - np.min(self.points[:, 1]))
        z_normalized = (self.points[:, 2] - np.min(self.points[:, 2])) / (np.max(self.points[:, 2]) - np.min(self.points[:, 2]))

        scaling_factor = 10000

        x_scaled = np.power(x_normalized, scaling_factor)
        y_scaled = np.power(y_normalized, scaling_factor)
        z_scaled = np.power(z_normalized, scaling_factor)

        # Stack the scaled values along the RGB axis (this will map each axis to one color channel)
        colors = np.stack([x_scaled, y_scaled, z_scaled], axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the denoised point cloud
        # o3d.visualization.draw_geometries([pcd])


        vis = o3d.visualization.Visualizer()
        vis.create_window()

        #Set smaller points
        vis.get_render_option().point_size = 1

        # Add geometry and render
        vis.add_geometry(pcd)

        # Run visualization
        vis.run()
        vis.destroy_window()

    def scaleMtrx(self, scale):
        #Gets the matrix for scaling
        #Input: [x,y,z] where each is the scaling factor in each direction

        #Scales the points by scale = [x,y,z]
        scaleMx = np.array([
            [scale[0], 0, 0, 0],  
            [0, scale[1], 0, 0],  
            [0, 0, scale[2], 0],  
            [0, 0, 0, 1]      
        ])
        return scaleMx

    def rotxMtrx(self, angle):
        #Gets the matrix for rotating the points around the x axis
        #Input: angle (in degrees) of desired rotation

        sinX = math.sin(math.radians(angle))
        cosX = math.cos(math.radians(angle))

        rotxMx = np.array([
            [1, 0, 0, 0],  
            [0, cosX, -sinX, 0],  
            [0, sinX, cosX, 0],  
            [0, 0, 0, 1]      
        ])

        return rotxMx

    def rotyMtrx(self, angle):
        #Gets the matrix for rotating the points around the y axis
        #Input: angle (in degrees) of desired rotation

        sinY = math.sin(math.radians(angle))
        cosY = math.cos(math.radians(angle))

        rotyMx = np.array([
            [cosY, 0, sinY, 0],  
            [0, 1, 0, 0],  
            [-sinY, 0, cosY, 0],  
            [0, 0, 0, 1]      
        ])

        return rotyMx

    def rotzMtrx(self, angle):
        #Gets the matrix for rotating the points around the z axis
        #Input: angle (in degrees) of desired rotation

        sinZ = math.sin(math.radians(angle))
        cosZ = math.cos(math.radians(angle))

        rotzMx = np.array([
            [cosZ, -sinZ, 0, 0],  
            [sinZ, cosZ, 0, 0],  
            [0, 0, 1, 0],  
            [0, 0, 0, 1]      
        ])

        return rotzMx

    def transMtrx(self, translate):
        #Gets the matrix for translation in x,y,z direction
        #Input: translate = [x,y,z] translation in cooresponding direction
        translate = np.array([
            [1, 0, 0, translate[0]],  
            [0, 1, 0, translate[1]],  
            [0, 0, 1, translate[2]],  
            [0, 0, 0, 1]      
        ])

        return translate

    def scalePnts(self, scale):
        #Scales the points by [x,y,z] where each value is the scale in cooresponding axis

        ones_column = np.ones((self.points.shape[0], 1))
        matrix_Nx4 = np.hstack([self.points, ones_column])

        s = self.scaleMtrx(scale)

        self.points = (matrix_Nx4 @ s.T)[:, :-1]

    def rotateXPnts(self, angle):
        #Rotates the points around x axis by angle number of degrees

        ones_column = np.ones((self.points.shape[0], 1))
        matrix_Nx4 = np.hstack([self.points, ones_column])

        rx = self.rotxMtrx(angle)

        self.points = (matrix_Nx4 @ rx.T)[:, :-1]

    def rotateYPnts(self, angle):
        #Rotates the points around y axis by angle number of degrees

        ones_column = np.ones((self.points.shape[0], 1))
        matrix_Nx4 = np.hstack([self.points, ones_column])

        ry = self.rotyMtrx(angle)

        self.points = (matrix_Nx4 @ ry.T)[:, :-1]

    def rotateZPnts(self, angle):
        #Rotates the points around z axis by angle number of degrees

        ones_column = np.ones((self.points.shape[0], 1))
        matrix_Nx4 = np.hstack([self.points, ones_column])

        rz = self.rotzMtrx(angle)

        self.points = (matrix_Nx4 @ rz.T)[:, :-1]

    def translatePnts(self, trans):
        #Translates the points 
        #Input: translate = [x,y,z] translation in cooresponding direction

        ones_column = np.ones((self.points.shape[0], 1))
        matrix_Nx4 = np.hstack([self.points, ones_column])

        t = self.transMtrx(trans)

        self.points = (matrix_Nx4 @ t.T)[:, :-1]


    def transform(self, scale, rot, translate):
        #Transform a set of 3d points
        #Scale scales the points by scale = [x,y,z]
        #rot rotates the points around the x,y,z axis by [x,y,z] degrees
        #translate translates the points in x,y,z direction by [x,y,z]
        #JUST ONE POSSIBLE ORDER OF THE TRANSFORMATIONS, ORDER MATTERS

        self.scalePnts(scale)
        self.rotateXPnts(rot[0])
        self.rotateYPnts(rot[1])
        self.rotateZPnts(rot[2])
        self.translatePnts(translate)

    def denoiseGaussFilter(self, sigma=1):
        #Denoises the lidar points
        #Sigma determines how strong the denoising is 

        # Apply Gaussian filter along each column (x, y, z separately)
        denoised = np.zeros_like(self.points)
        for i in range(3):  # x, y, z
            denoised[:, i] = gaussian_filter1d(self.points[:, i], sigma=sigma)

        self.points = denoised
    
    def denoiseNonLocal(self, h=.1, search_radius=.5, k=20):
        """
        Non-local means denoising for 3D point clouds.
        
        Args:
            h (float): Filtering parameter controlling smoothing strength.
            search_radius (float): Radius to search for similar points.
            k (int): Max number of neighbors to consider.

        Returns:
            np.ndarray: Denoised points.
        """
        
        nbrs = NearestNeighbors(radius=search_radius).fit(self.points)
        denoised_points = np.zeros_like(self.points)
        
        for i, p in enumerate(self.points):
            indices = nbrs.radius_neighbors([p], return_distance=False)[0]
            
            if len(indices) == 0:
                denoised_points[i] = p
                continue
            
            neighbors = self.points[indices]
            
            # Compute distances
            dists = np.linalg.norm(neighbors - p, axis=1)
            
            # Compute weights based on distance
            weights = np.exp(-(dists**2) / (h**2))
            weights /= np.sum(weights)
            
            # Weighted average
            denoised_points[i] = np.sum(neighbors * weights[:, None], axis=0)
            
        return denoised_points

def add_gaussian_noise(data, mean=0, std_dev=2):
    #Add noise to illustrate denoising. 
    #DO NOT USE IN NORMAL USECASE, WILL MAKE DATA WORSE
    noise = np.random.normal(loc=mean, scale=std_dev, size=data.shape)
    return data + noise

def visualizeThree(noisy, denoised, original):
    #Use to visualize three point clouds side by side

    #original
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original)

    #denoised
    pcd_denoise = o3d.geometry.PointCloud()
    pcd_denoise.points = o3d.utility.Vector3dVector(denoised)

    #noisy
    pcd_noisy = o3d.geometry.PointCloud()
    pcd_noisy.points = o3d.utility.Vector3dVector(noisy)

    # Translate denoised point cloud along x-axis to separate the two clouds visually
    translation_distance = (np.max(noisy[:, 0]) - np.min(noisy[:, 0])) * 1.2
    pcd_noisy.translate((translation_distance, 0, 0))

    pcd_original.translate((-translation_distance, 0, 0))

    # # Colors for original point cloud (xyz normalized)
    # def get_colors(points):
    #     x_norm = (points[:, 0] - np.min(points[:, 0])) / (np.max(points[:, 0]) - np.min(points[:, 0]))
    #     y_norm = (points[:, 1] - np.min(points[:, 1])) / (np.max(points[:, 1]) - np.min(points[:, 1]))
    #     z_norm = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))
        
    #     scaling_factor = 10000
    #     x_scaled = np.power(x_norm, scaling_factor)
    #     y_scaled = np.power(y_norm, scaling_factor)
    #     z_scaled = np.power(z_norm, scaling_factor)
        
    #     return np.stack([x_scaled, y_scaled, z_scaled], axis=-1)

    # pcd_original.colors = o3d.utility.Vector3dVector(get_colors(noisy))
    # pcd_denoise.colors = o3d.utility.Vector3dVector(get_colors(np.asarray(denoised)))

    # Visualizer setup

    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1

    vis.add_geometry(pcd_original)
    vis.add_geometry(pcd_noisy)
    vis.add_geometry(pcd_denoise)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    #Example usecase 

    #This just showcases that i can use a file or array of points to make the lazPnts object
    obj = lazPnts("Palac_Moszna.laz")
    points = obj.getPnts()
    obj2 = lazPnts(points)

    #Apply transformation and visualize. Play around with it!
    scale = [1,1,1] #Scale in [x,y,z] direction
    rot = [0,0,0] #Rotate around [x,y,z] axis
    trans = [0,0,0] #Translate in [x,y,z] direction
    obj2.transform(scale, rot, trans)
    obj2.visualize()

    # #add noisy data to demonstrate denoising
    # noisyData = add_gaussian_noise(obj2.getPnts())
    # noisyObj = lazPnts(noisyData)

    # #Demonstrate it after denoising
    # #noisyObj.denoiseGaussFilter()
    # visualizeThree(noisyData, noisyObj.getPnts(), points)


    # noisyObj.visualize()





