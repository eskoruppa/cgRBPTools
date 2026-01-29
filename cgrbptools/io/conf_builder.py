from __future__ import annotations

import numpy as np

from ..SO3 import so3
from .lmp_topol import CGRBPTopology
from .lmp_conf import CGRBPConf


    
class ConfBuilder:
    
    mapping = {
        "circ": "circular",
        "circular": "circular",
        "closed": "circular",
        "str": "straight",
        "straight": "straight",
        "linear": "straight",
        "line": "straight",
        "shape": "ground_state",
        "ground_state": "ground_state",
        "groundstate": "ground_state",
        "gs": "ground_state",
    }
    
    def __init__(self, topology: CGRBPTopology):
        self.topology = topology
     
    @classmethod
    def mapping_dict(cls) -> dict[str, str]:
        """
        Get the mapping dictionary of configuration types to method names.
        
        Returns
        -------
        dict[str, str]
            Dictionary mapping configuration type strings to method names.
        """
        return cls.mapping
    
    @classmethod
    def build(
        cls,
        topology: CGRBPTopology,
        conf_type: str,
        mass: float = 1,
        verbose = False,
    ) -> CGRBPConf:
        """
        Build a CGRBPConf of the specified type using the stored topology.
        
        Parameters
        ----------
        conf_type : str
            Type of configuration to build. Supported types are:
            - "straight" or "str": Straight chain configuration.
            - "circular" or "circ": Circular configuration.
            - "shape": (Not implemented in this method)
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        
        Returns
        -------
        CGRBPConf
            Generated configuration object.
        
        Raises
        ------
        ValueError
            If conf_type is not recognized.
        
        """
        conf_type_lower = conf_type.lower()
        if conf_type_lower not in cls.mapping:
            raise ValueError(f"Unsupported conf_type '{conf_type}'. Supported types are: {list(cls.mapping.keys())}")
        
        method_name = cls.mapping[conf_type_lower]
        method = getattr(cls, method_name)
        if verbose:
            print(f"Generating configuration using method: {method_name}.")
        return method(topology, mass=mass)
        


    @classmethod
    def straight(
        cls,
        topology: CGRBPTopology,
        excess_twist: float = 0,
        mass: float = 1,
        orientation: np.ndarray | list | tuple = np.array([0.0, 0., 1.]),
        origin: np.ndarray | list | tuple = np.zeros(3),
    ) -> CGRBPConf:
        """
        Generate a straight DNA configuration with optional excess twist.
        
        Constructs a linear chain of rigid base pairs starting from the origin
        with the specified orientation. The configuration is built by applying
        successive SE3 transformations from the topology's groundstate, with
        optional additional twist.
        
        Parameters
        ----------
        topology : CGRBPTopology
            Topology object containing groundstate parameters and stiffness.
            Must have groundstate initialized via set_params().
        excess_twist : float, optional
            Total excess twist in radians to distribute uniformly across all steps.
            Default is 0 (no excess twist).
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        orientation : array-like of shape (3,), optional
            Direction vector for the initial base pair orientation. Will be normalized.
            Default is [0, 0, 1] (along z-axis).
        origin : array-like of shape (3,), optional
            Starting position of the first base pair in 3D space.
            Default is [0, 0, 0].
        
        Returns
        -------
        CGRBPConf
            Configuration object containing poses (4x4 SE3 matrices) and mass
            for all base pairs in the chain.
        
        Raises
        ------
        TypeError
            If topology is not CGRBPTopology, or if orientation/origin cannot
            be converted to float arrays.
        ValueError
            If topology.groundstate is None, topology.nbp <= 0, mass <= 0,
            orientation is a zero vector, or orientation/origin have wrong shape.
        
        """
        # Validate topology
        if not isinstance(topology, CGRBPTopology):
            raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
        if topology.groundstate is None:
            raise ValueError("topology.groundstate is not set. Call topology.set_params() first.")
        if topology.nbp <= 0:
            raise ValueError(f"topology.nbp must be positive, got {topology.nbp}")
        
        # Validate mass
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError(f"mass must be a positive number, got {mass}")
        
        # Validate and convert orientation to numpy array
        try:
            orientation = np.asarray(orientation, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"orientation must be array-like and convertible to float: {e}")
        if orientation.shape != (3,):
            raise ValueError(f"orientation must be a 3D vector, got shape {orientation.shape}")
        orientation_norm = np.linalg.norm(orientation)
        if orientation_norm < 1e-10:
            raise ValueError("orientation must be a non-zero vector")
        
        # Validate and convert origin to numpy array
        try:
            origin = np.asarray(origin, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"origin must be array-like and convertible to float: {e}")
        if origin.shape != (3,):
            raise ValueError(f"origin must be a 3D vector, got shape {origin.shape}")
        
        nbp = topology.nbp
        poses = np.zeros((nbp,4,4),dtype=float)
        poses[0] = np.eye(4)
        poses[0][:3,3] = origin
        if not np.allclose(orientation, np.array([0,0,1])):
            R = so3.rotmat_align_vector(np.array([0,0,1]), orientation)
            poses[0][:3,:3] = R
        
        # For closed chains, the last groundstate entry connects back to the first element
        # so we exclude it from the forward chain construction
        num_steps = topology.nbps - 1 if topology.closed else topology.nbps
        excess_twist_per_step = excess_twist / topology.nbps
        for i in range(num_steps):
            X0 = topology.groundstate[i]
            Xstr = np.zeros(X0.shape)
            Xstr[2] = X0[2] + excess_twist_per_step
            Xstr[5] = X0[5]
            g = so3.se3_euler2rotmat(Xstr)
            poses[i+1] = poses[i] @ g
        conf = CGRBPConf(poses, mass)
        conf.set_topology(topology)
        return conf
    
    @classmethod
    def circular(
        cls,
        topology: CGRBPTopology,
        excess_link: float = 0.0,
        mass: float = 1.0,
        normal: np.ndarray | list | tuple = np.array([1.0, 0.0, 0.0]),
        center: np.ndarray | list | tuple = np.zeros(3),
    ) -> CGRBPConf:
        """
        Generate a circular DNA configuration.
        
        Constructs a regular polygon arrangement of rigid base pairs in the plane
        perpendicular to the normal vector, centered at the specified center point.
        The polygon is built by successive rotations around the normal axis, with
        each step including both rotation and translation (rise). Twist is then
        applied to achieve integer linking number.
        
        Parameters
        ----------
        topology : CGRBPTopology
            Topology object containing groundstate parameters and stiffness.
            Must have groundstate initialized via set_params().
        excess_link : float, optional
            Additional linking number beyond the rounded natural linking from
            groundstate. The total linking number will be round(total_twist/(2π)) +
            excess_link. Default is 0.
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        normal : array-like of shape (3,), optional
            Normal vector perpendicular to the plane of the polygon. Will be
            normalized. Default is [1, 0, 0] (polygon in yz-plane).
        center : array-like of shape (3,), optional
            Center position of the polygon in 3D space.
            Default is [0, 0, 0].
        
        Returns
        -------
        CGRBPConf
            Configuration object containing poses (4x4 SE3 matrices) and mass
            for all base pairs arranged in a polygon.
        
        Raises
        ------
        TypeError
            If topology is not CGRBPTopology, or if normal/center cannot
            be converted to float arrays.
        ValueError
            If topology.groundstate is None, topology.nbp <= 0, mass <= 0,
            normal is a zero vector, or normal/center have wrong shape.
        """
        # Validate topology
        if not isinstance(topology, CGRBPTopology):
            raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
        if topology.groundstate is None:
            raise ValueError("topology.groundstate is not set. Call topology.set_params() first.")
        if topology.nbp <= 0:
            raise ValueError(f"topology.nbp must be positive, got {topology.nbp}")
        
        # Validate mass
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError(f"mass must be a positive number, got {mass}")
        
        # Validate and convert normal to numpy array
        try:
            normal = np.asarray(normal, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"normal must be array-like and convertible to float: {e}")
        if normal.shape != (3,):
            raise ValueError(f"normal must be a 3D vector, got shape {normal.shape}")
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            raise ValueError("normal must be a non-zero vector")
        normal = normal / normal_norm  # Normalize
        
        # Validate and convert center to numpy array
        try:
            center = np.asarray(center, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"center must be array-like and convertible to float: {e}")
        if center.shape != (3,):
            raise ValueError(f"center must be a 3D vector, got shape {center.shape}")
        
        # Calculate average rise from groundstate
        avg_rise = np.mean(topology.groundstate[:, 5])
        
        nbp = topology.nbp
        X0 = topology.groundstate
        
        poses = np.zeros((nbp,4,4),dtype=float)
        poses[0] = np.eye(4)
        
        # generate circle
        nn = np.array([0.0,1.0,0.0])
        R = so3.euler2rotmat(nn*2*np.pi/nbp) 
        Rh = so3.euler2rotmat(nn*np.pi/nbp) 
        for i in range(1,nbp):
            poses[i,:3,3] = poses[i-1,:3,3] + Rh @ poses[i-1,:3,2] * avg_rise
            poses[i,:3,:3] = R @ poses[i-1,:3,:3]
            poses[i,3,3] = 1.0
            
        # calculate twist
        total_twist = np.sum(X0[:,2])
        if not topology.closed:
            total_twist +=  np.mean(X0[:,2])
        
        rounded_twist = round(total_twist / (2 * np.pi) + excess_link) * 2 * np.pi
        excess_twist = rounded_twist - total_twist
        excess_twist_per_step = excess_twist / nbp
        
        accu_twist = 0
        for i in range(1, nbp):
            # Get twist for junction (i-1)→i
            step_twist = X0[i - 1, 2] + excess_twist_per_step
            accu_twist += step_twist
            
            # Apply rotation around the tangent (third column)
            R = so3.euler2rotmat(poses[i, :3, 2] * accu_twist)
            poses[i, :3, :3] = R @ poses[i, :3, :3]

        # shift to origin
        shift_to_mean = - np.mean(poses[:,:3,3],axis=0)
        for i in range(nbp):
            poses[i][:3,3] += shift_to_mean
            
        # rotate circle to align normals 
        normal = normal / np.linalg.norm(normal)
        R_align = so3.rotmat_align_vector(np.array([0.0, 0.0, 1.0]), normal)
        for i in range(len(poses)):
            poses[i][:3,:3] = R_align @ poses[i][:3,:3]
            poses[i][:3,3] = R_align @ poses[i][:3,3] + center
        
        # check determinants
        for i,pose in enumerate(poses):
            det = np.linalg.det(pose[:3,:3])
            if not np.isclose(det,1.0):
                raise ValueError(f"Rotation matrix at index {i} has invalid determinant {det}")

        conf = CGRBPConf(poses, mass)
        conf.set_topology(topology)
        return conf

    @classmethod
    def ground_state(
        cls,
        topology: CGRBPTopology,
        mass: float = 1,
        orientation: np.ndarray | list | tuple = np.array([0, 0, 1]),
        origin: np.ndarray | list | tuple = np.zeros(3),
    ) -> CGRBPConf:
        """
        Generate a DNA configuration using groundstate parameters without modifications.
        
        Creates an OPEN chain configuration by applying the SE3 transformations from
        the topology's groundstate exactly as provided, without any excess twist.
        For closed topologies, the last groundstate entry (which connects back to
        the first element) is excluded to create an open representation.
        
        Parameters
        ----------
        topology : CGRBPTopology
            Topology object containing groundstate parameters.
            Must have groundstate initialized via set_params().
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        orientation : array-like of shape (3,), optional
            Direction vector for the initial base pair orientation.
            Default is [0, 0, 1] (along z-axis).
        origin : array-like of shape (3,), optional
            Starting position of the first base pair in 3D space.
            Default is [0, 0, 0].
        
        Returns
        -------
        CGRBPConf
            Configuration object containing poses (4x4 SE3 matrices) and mass
            for all base pairs in an open chain.
        
        Raises
        ------
        TypeError
            If topology is not CGRBPTopology, or if orientation/origin cannot
            be converted to float arrays.
        ValueError
            If topology.groundstate is None, topology.nbp <= 0, mass <= 0,
            orientation is a zero vector, or orientation/origin have wrong shape.
        
        """
        # Validate topology
        if not isinstance(topology, CGRBPTopology):
            raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
        if topology.groundstate is None:
            raise ValueError("topology.groundstate is not set. Call topology.set_params() first.")
        if topology.nbp <= 0:
            raise ValueError(f"topology.nbp must be positive, got {topology.nbp}")
        
        # Validate mass
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError(f"mass must be a positive number, got {mass}")
        
        # Validate and convert orientation to numpy array
        try:
            orientation = np.asarray(orientation, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"orientation must be array-like and convertible to float: {e}")
        if orientation.shape != (3,):
            raise ValueError(f"orientation must be a 3D vector, got shape {orientation.shape}")
        orientation_norm = np.linalg.norm(orientation)
        if orientation_norm < 1e-10:
            raise ValueError("orientation must be a non-zero vector")
        
        # Validate and convert origin to numpy array
        try:
            origin = np.asarray(origin, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"origin must be array-like and convertible to float: {e}")
        if origin.shape != (3,):
            raise ValueError(f"origin must be a 3D vector, got shape {origin.shape}")
        
        # For closed topologies, exclude the last groundstate entry (which connects back to first)
        # to create an open chain representation
        if topology.closed:
            tX0 = topology.groundstate[:-1]
        else:
            tX0 = topology.groundstate
        
        nbps = len(tX0)
        nbp = nbps + 1
            
        poses = np.zeros((nbp, 4, 4), dtype=float)
        poses[0] = np.eye(4)
        poses[0][:3, 3] = origin
        if not np.allclose(orientation, np.array([0, 0, 1])):
            R = so3.rotmat_align_vector(np.array([0, 0, 1]), orientation)
            poses[0][:3, :3] = R
        
        for i in range(1, nbp):
            g = so3.se3_euler2rotmat(tX0[i - 1])
            poses[i] = poses[i - 1] @ g
        
        conf = CGRBPConf(poses, mass)
        conf.set_topology(topology)
        return conf
     
    @classmethod
    def from_poses(
        cls,
        poses: np.ndarray,
        topology: CGRBPTopology | None = None,
        mass: float = 1,
    ) -> CGRBPConf:
        """
        Create a DNA configuration from SE3 poses (transformation matrices).
        
        Constructs a CGRBPConf directly from an array of 4x4 SE3 transformation
        matrices representing the pose (position and orientation) of each base pair.
        
        Parameters
        ----------
        poses : np.ndarray
            Array of SE3 transformation matrices with shape (nbp, 4, 4).
            Each 4x4 matrix should be a valid SE3 transformation with:
            - Upper-left 3x3 block: rotation matrix
            - Upper-right 3x1 block: translation vector
            - Bottom row: [0, 0, 0, 1]
        topology : CGRBPTopology, optional
            Optional topology object. If provided, the number of poses must
            match topology.nbp. Default is None.
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        
        Returns
        -------
        CGRBPConf
            Configuration object containing the provided poses and mass.
        
        Raises
        ------
        TypeError
            If poses is not a numpy array or topology is not CGRBPTopology.
        ValueError
            If poses shape is invalid, mass <= 0, nbp <= 0, or number of poses
            doesn't match topology.nbp when topology is provided.
        
        Notes
        -----
        - No validation is performed on whether rotation matrices are orthogonal
        - Bottom row of SE3 matrices is not validated
        - If topology is provided, it will be attached to the configuration
        """
        # Validate poses
        if not isinstance(poses, np.ndarray):
            raise TypeError(f"poses must be a numpy array, got {type(poses)}")
        
        if poses.ndim != 3:
            raise ValueError(f"poses must be 3-dimensional with shape (nbp, 4, 4), got {poses.ndim} dimensions")
        
        if poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must have shape (nbp, 4, 4), got {poses.shape}")
        
        nbp = poses.shape[0]
        if nbp <= 0:
            raise ValueError(f"Number of poses must be positive, got {nbp}")
        
        # Validate mass
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError(f"mass must be a positive number, got {mass}")
        
        # Validate topology if provided
        if topology is not None:
            if not isinstance(topology, CGRBPTopology):
                raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
            if nbp != topology.nbp:
                raise ValueError(
                    f"Number of poses ({nbp}) does not match topology.nbp ({topology.nbp})"
                )
        
        conf = CGRBPConf(poses, mass)
        if topology is not None:
            conf.set_topology(topology)
        return conf
    
    @classmethod
    def from_positions_orientations(
        cls,
        positions: np.ndarray,
        triads: np.ndarray | None = None,
        quaternions: np.ndarray | None = None,
        topology: CGRBPTopology | None = None,
        mass: float = 1,
    ) -> CGRBPConf:
        """
        Create a DNA configuration from positions and orientations.
        
        Constructs a CGRBPConf from separate arrays of positions and orientations
        (provided as either rotation matrices or quaternions). This is useful when
        importing configurations from external sources.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of 3D positions with shape (nbp, 3).
        triads : np.ndarray, optional
            Array of rotation matrices (triads) with shape (nbp, 3, 3).
            If both triads and quaternions are provided, triads take precedence.
            Either triads or quaternions must be provided.
        quaternions : np.ndarray, optional
            Array of unit quaternions (normalized, magnitude 1) with shape (nbp, 4).
            Quaternions should be in [w, x, y, z] format as used by so3.quats2mats.
            Either triads or quaternions must be provided.
        topology : CGRBPTopology, optional
            Optional topology object. If provided, the number of positions must
            match topology.nbp. Default is None.
        mass : float, optional
            Mass of each base pair. Must be positive. Default is 1.
        
        Returns
        -------
        CGRBPConf
            Configuration object containing the constructed poses and mass.
        
        Raises
        ------
        TypeError
            If positions/triads/quaternions are not numpy arrays, or topology
            is not CGRBPTopology.
        ValueError
            If neither triads nor quaternions are provided, shapes are invalid,
            mass <= 0, nbp <= 0, number of positions/orientations don't match,
            or nbp doesn't match topology.nbp when topology is provided.
        
        Warnings
        --------
        UserWarning
            If both triads and quaternions are provided (triads will be used).
        
        Notes
        -----
        - When both triads and quaternions are provided, triads take precedence
        - No validation is performed on whether rotation matrices are orthogonal
        - If topology is provided, it will be attached to the configuration
        """
        # Validate that at least one orientation is provided
        if triads is None and quaternions is None:
            raise ValueError("Either triads or quaternions must be provided")
        
        # Warn if both are provided
        if triads is not None and quaternions is not None:
            import warnings
            warnings.warn(
                "Both triads and quaternions provided. Using triads and ignoring quaternions.",
                UserWarning
            )
        
        # Validate positions
        if not isinstance(positions, np.ndarray):
            raise TypeError(f"positions must be a numpy array, got {type(positions)}")
        
        if positions.ndim != 2:
            raise ValueError(f"positions must be 2-dimensional with shape (nbp, 3), got {positions.ndim} dimensions")
        
        if positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (nbp, 3), got {positions.shape}")
        
        nbp = positions.shape[0]
        if nbp <= 0:
            raise ValueError(f"Number of positions must be positive, got {nbp}")
        
        # Validate and process orientations
        if triads is not None:
            if not isinstance(triads, np.ndarray):
                raise TypeError(f"triads must be a numpy array, got {type(triads)}")
            
            if triads.ndim != 3:
                raise ValueError(f"triads must be 3-dimensional with shape (nbp, 3, 3), got {triads.ndim} dimensions")
            
            if triads.shape != (nbp, 3, 3):
                raise ValueError(f"triads must have shape ({nbp}, 3, 3), got {triads.shape}")
            
            orientations = triads
        else:
            # Use quaternions
            if not isinstance(quaternions, np.ndarray):
                raise TypeError(f"quaternions must be a numpy array, got {type(quaternions)}")
            
            if quaternions.ndim != 2:
                raise ValueError(f"quaternions must be 2-dimensional with shape (nbp, 4), got {quaternions.ndim} dimensions")
            
            if quaternions.shape != (nbp, 4):
                raise ValueError(f"quaternions must have shape ({nbp}, 4), got {quaternions.shape}")
            
            # Validate that quaternions are unit quaternions (magnitude ~1)
            quat_norms = np.linalg.norm(quaternions, axis=1)
            if not np.allclose(quat_norms, 1.0, atol=1e-6):
                max_deviation = np.max(np.abs(quat_norms - 1.0))
                raise ValueError(
                    f"quaternions must be unit quaternions (magnitude 1). "
                    f"Maximum deviation from unit norm: {max_deviation:.6e}"
                )
            
            # Convert quaternions to rotation matrices
            orientations = so3.quats2mats(quaternions)
        
        # Validate mass
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError(f"mass must be a positive number, got {mass}")
        
        # Validate topology if provided
        if topology is not None:
            if not isinstance(topology, CGRBPTopology):
                raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
            if nbp != topology.nbp:
                raise ValueError(
                    f"Number of positions ({nbp}) does not match topology.nbp ({topology.nbp})"
                )
        
        # Construct poses
        poses = np.zeros((nbp, 4, 4), dtype=float)
        poses[:, :3, 3] = positions
        poses[:, :3, :3] = orientations
        poses[:, 3, 3] = 1.0
        
        conf = CGRBPConf(poses, mass)
        if topology is not None:
            conf.set_topology(topology)
        return conf
        