from __future__ import annotations

from pathlib import Path

import numpy as np

from ..SO3 import so3
from .topology import CGRBPTopology
from .configuration import CGRBPConf
from .conf_import import (
    ConfigurationMismatchError,
    load_poses,
    positions_to_nm,
    remove_repeated_pose,
    validate_poses,
)

class ConfBuilder:
    
    mapping = {
        "circ": "circular",
        "circular": "circular",
        "circle": "circular",
        "closed": "circular",
        "str": "straight",
        "straight": "straight",
        "linear": "straight",
        "line": "straight",
        "lin": "straight",
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
        excess_link: float = 0.0,
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
        
        if method_name in ["straight", "circular"]:
            return method(topology, excess_link=excess_link, mass=mass)
        return method(topology, mass=mass)
        


    @classmethod
    def straight(
        cls,
        topology: CGRBPTopology,
        excess_link: float = 0,
        mass: float = 1,
        orientation: np.ndarray | list | tuple = np.array([0.0, 0., 1.]),
        origin: np.ndarray | list | tuple = np.zeros(3),
    ) -> CGRBPConf:
        """
        Generate a straight DNA configuration with optional excess link.
        
        Constructs a linear chain of rigid base pairs starting from the origin
        with the specified orientation. The configuration is built by applying
        successive SE3 transformations from the topology's groundstate, with
        optional additional link.
        
        Parameters
        ----------
        topology : CGRBPTopology
            Topology object containing groundstate parameters and stiffness.
            Must have groundstate initialized via set_params().
        excess_link : float, optional
            Total excess link to distribute uniformly across all steps.
            Default is 0 (no excess link).
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
        excess_twist_per_step = 2*np.pi * excess_link / topology.nbps
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
        validate: bool = True,
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
        validate : bool, optional
            Check that the poses are SE(3) elements (see conf_import.validate_poses) and
            re-orthonormalize rotation blocks with small deviations. Default is False.

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
        ConfigurationValidationError
            If validate is set and the poses are not valid SE(3) elements.

        Notes
        -----
        - Without validate, neither the orthogonality of the rotation matrices nor the
          bottom row of the SE3 matrices is validated
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

        if validate:
            poses, _ = validate_poses(poses)

        conf = CGRBPConf(poses, mass)
        if topology is not None:
            conf.set_topology(topology)
        return conf

    @classmethod
    def from_file(
        cls,
        topology: CGRBPTopology,
        source: str | Path | np.ndarray,
        frame: int | None = None,
        mass: float = 1,
        conf_units: str = 'nm',
        force_orthogonalize: bool = False,
    ) -> CGRBPConf:
        """
        Create a configuration from an external configuration (file or array).

        The configuration is loaded (see conf_import.load_poses), converted to the length unit
        of the topology, validated, and matched to the beads of the topology. It may be given at
        bead resolution (topology.nbp poses) or at base-pair resolution (one pose per base pair
        of the topology's sequence), in which case the frames of the beads are retained. A
        closed configuration that repeats its first pose at the end has the repeated pose
        removed. The number of poses has to match exactly; truncation is only available in
        lmp_input, where the sequence can still be adapted.

        Parameters
        ----------
        topology : CGRBPTopology
            Topology with parameters and sequence set.
        source : str, Path or np.ndarray
            Configuration array or path to a .npy or .npz file.
        frame : int, optional
            Snapshot to select if the source holds a trajectory (-1 selects the last one).
        mass : float, optional
            Mass of each bead. Default is 1.
        conf_units : str, optional
            Length unit of the positions: 'nm' (default), 'angstrom' or 'sim' (multiples of
            topology.unit_length).
        force_orthogonalize : bool, optional
            Re-orthonormalize rotation blocks that deviate by more than the tolerance.

        Returns
        -------
        CGRBPConf
            Configuration with positions in the length unit of the topology.

        Raises
        ------
        ConfigurationError
            If the configuration cannot be loaded, is invalid or does not match the topology.
        """
        if not isinstance(topology, CGRBPTopology):
            raise TypeError(f"topology must be a CGRBPTopology instance, got {type(topology)}")
        name = f"Configuration file '{source}'" if not isinstance(source, np.ndarray) else 'The configuration array'
        poses = positions_to_nm(load_poses(source, frame=frame), conf_units, topology.unit_length)
        poses, _ = validate_poses(poses, force_orthogonalize=force_orthogonalize, name=name)
        if topology.closed:
            poses, _ = remove_repeated_pose(poses, name=name)

        sequence = topology.sequence
        if len(poses) != topology.nbp:
            cg = topology.composite_size if topology.seqs_set else 1
            if sequence is None or cg == 1 or len(poses) != len(sequence):
                raise ConfigurationMismatchError(
                    f'Configuration mismatch: {name[0].lower() + name[1:]} contains {len(poses)} '
                    f'poses, but the topology requires {topology.nbp} poses at bead resolution'
                    + (f' or {len(sequence)} poses at base-pair resolution.' if sequence and cg > 1 else '.')
                )
            poses = poses[topology.center_pos::cg][:topology.nbp]

        poses[:, :3, 3] /= topology.unit_length
        return cls.from_poses(poses, topology, mass=mass)
    
    @classmethod
    def from_tracepoints(
        cls,
        topology: CGRBPTopology,
        tracepoints: np.ndarray,
        *,
        excess_link: float | None = None,
        link_reference: str = 'lk0',
        mass: float = 1,
        conf_units: str = 'nm',
        rescale: bool = False,
        max_fene: float | str | None = 'auto',
        tangent_persistence: float | np.ndarray = 1.0,
        smoothing: float | None = None,
        first_triad: np.ndarray | None = None,
        twist_correction: bool = True,
        return_info: bool = False,
    ) -> CGRBPConf | tuple:
        """
        Generate a DNA configuration along a smooth curve through tracepoints.

        The beads are placed along a cubic spline through the tracepoints and carry the intrinsic
        twist and rise of the topology's groundstate; tilt, roll, shift and slide are ignored (see
        tracepoints.tracepoints_to_poses). The number of beads and whether the chain is closed
        follow from the topology. Consecutive beads are spaced in proportion to the rise: without
        rescale all steps are stretched or compressed by the same factor, the first bead sits at
        the first tracepoint (for closed curves with smoothing, at its smoothed position) and, for
        open chains, the last bead at the last one; with rescale the curve is scaled about the
        centroid of the tracepoints such that the distances equal the rise. For coarse-grained
        topologies the curve is the path of the beads, i.e. of the base pairs
        k*composite_size + center_pos.

        The excess link adds twist on top of the intrinsic twist:

        - open chains: 2 pi excess_link / nbps of twist per step, as in ConfBuilder.straight.
        - closed chains, link_reference='lk0' (default): the linking number becomes
          round(Lk0 + excess_link), with Lk0 the total intrinsic twist in turns (the twist of
          every step taken in [-pi, pi]), as for ConfBuilder.circular. The twist then also
          compensates the writhe of the path. Unless the path is planar, the linking number of
          the ring is evaluated with PyLk, in a time quadratic in the number of beads (about
          0.4 s for 1,000 and 25 s for 8,000 beads).
        - closed chains, link_reference='path': excess_link whole turns are added to the
          twist-relaxed ring for this path, as in tracepoints_to_poses. Both references agree for
          gently bent planar paths; otherwise they differ by about Wr + E (see below).

        With excess_link=None (default) no twist is added. A closed chain then becomes the
        twist-relaxed ring for this path, closed by the smallest uniform change of the twist; its
        linking number is the integer closest to Lk0 + Wr + E, with Wr the writhe of the path and
        E the total turning of the bent steps about the tangents beyond their twist (negligible
        for gently bent paths, up to about pi*bend^2/8 per step).

        Parameters
        ----------
        topology : CGRBPTopology
            Topology with groundstate. Its number of beads, closed flag, unit length and FENE
            coefficients define the chain.
        tracepoints : array-like
            Positions with shape (N, 3), or positions and tangents with shape (N, 2, 3), in
            conf_units. Tangents may have any length; tangents marked NaN are free. A closed curve
            that repeats its first tracepoint at the end has the repetition removed.
        excess_link : float, optional
            Excess linking number (see above). Has to be an integer for closed chains with
            link_reference='path'. Default is None.
        link_reference : str, optional
            Reference of the excess link of closed chains, 'lk0' (default) or 'path' (see above).
            Ignored for open chains and without excess link.
        mass : float, optional
            Mass of each bead. Must be positive. Default is 1.
        conf_units : str, optional
            Length unit of the tracepoints, smoothing and max_fene, and of the lengths in the
            diagnostics and messages (except FENE coefficients and bond lengths in FENE messages,
            which are in simulation units): 'nm' (default), 'angstrom' or 'sim' (multiples of
            topology.unit_length).
        rescale : bool, optional
            Scale the curve such that the distances between consecutive beads equal the rise.
            Default is False.
        max_fene : float, 'auto' or None, optional
            Limit on stretched bonds. 'auto' (default): for topologies with FENE bonds no bond may
            exceed Rc + (R0 - Rc) * sqrt(1 - conf_checks.FENE_RLOGARG_MIN), the longest bond for
            which LAMMPS evaluates the FENE term regularly; no limit without FENE bonds. A number
            limits the stretch of every step beyond its rise (distance - rise, in conf_units), as
            in tracepoints_to_poses. None: no limit. Bonds in the FENE regime of the topology are
            reported as warnings in any case.
        tangent_persistence : float or np.ndarray, optional
            How far the curve follows a prescribed tangent before it turns, scalar or one value
            per tracepoint. Default is 1.
        smoothing : float, optional
            For noisy tracepoints: root-mean-square distance (conf_units) by which the curve may
            miss the tracepoints. Requires at least 4 tracepoints. Default is None.
        first_triad : np.ndarray, optional
            Triad (3x3, columns are the axes) of the first bead. Its third axis prescribes the
            tangent at the first tracepoint. Default is the triad of ConfBuilder.straight for the
            tangent of the curve.
        twist_correction : bool, optional
            Match the twist of every step exactly. Default is True.
        return_info : bool, optional
            Also return the diagnostics. Default is False.

        Returns
        -------
        CGRBPConf or tuple[CGRBPConf, TracepointInfo]
            Configuration with positions in simulation units (multiples of topology.unit_length),
            and the diagnostics if return_info is set, with lengths in conf_units. All warnings
            are issued as UserWarning and listed in TracepointInfo.warnings.

        Raises
        ------
        TypeError
            If topology is not a CGRBPTopology instance.
        ValueError
            If topology.groundstate is not set, an argument is invalid, the curve cannot be traced
            (see tracepoints_to_poses), the beads are spaced implausibly far from the rise
            without rescale, a bond exceeds the FENE limit, or the excess link adds more than 90
            degrees of twist per step. With max_fene='auto' also if the FENE coefficients of the
            topology are not available or its groundstate bonds already exceed the FENE limit.
        ImportError
            If the linking number of a non-planar ring is required (link_reference='lk0') and
            PyLk is not available.

        Notes
        -----
        With link_reference='lk0' the turns added to the twist-relaxed ring are
        TracepointInfo.excess_twist_per_step * nbp / (2 pi); the diagnostics describe the returned
        ring, except for the warning on the closure of the twist-relaxed ring. Steps whose twist
        would exceed 180 degrees in magnitude (composite steps with an intrinsic twist close to
        180 degrees, e.g. 5 base pairs per bead) appear twisted the other way round in the
        rotation-vector convention, although
        their deformation relative to the groundstate carries the intended twist; this is
        reported as a warning. The elastic energy of the configuration can be assessed with
        conf_checks.energy_check.

        lmp_input -dlk (conf_checks.linking_number) counts the linking number from the twist of
        the junctions relative to the full groundstate, whereas link_reference='lk0' counts the
        linking number of the ribbon along the bead frames, as for ConfBuilder.circular. The two
        counts differ by an offset that grows with the number of beads, most at composite sizes
        of a few base pairs per bead. For long rings -dlk may therefore report the result as one
        or more turns off (see KNOWN_ISSUES.md).
        """
        import warnings

        from .conf_checks import (
            CONF_LK_INTEGER_TOL,
            CONF_MAX_ADDED_TWIST,
            FENE_RLOGARG_MIN,
            bond_vectors,
            check_fene,
        )
        from .conf_import import CONF_UNITS, CONF_UNITS_TO_NM, _listed
        from .topology import LMP_TOPOL_GROUNDSTATE_MAX_FACTOR, LMP_TOPOL_GROUNDSTATE_MIN_FACTOR
        from .tracepoints import TRACE_FENE_TOL, TRACE_INTEGER_TOL, tracepoints_to_poses

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

        # Validate options
        if conf_units not in CONF_UNITS:
            raise ValueError(
                f"Unknown length unit '{conf_units}'. Expected one of: {', '.join(CONF_UNITS)}."
            )
        if link_reference not in ('lk0', 'path'):
            raise ValueError(f"link_reference must be 'lk0' or 'path', got {link_reference!r}.")
        auto_fene = isinstance(max_fene, str)
        if auto_fene and max_fene != 'auto':
            raise ValueError(f"max_fene must be a number, 'auto' or None, got {max_fene!r}.")
        closed = topology.closed
        if excess_link is not None:
            if isinstance(excess_link, bool) or not isinstance(
                excess_link, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"excess_link must be a finite number or None, got {excess_link!r}."
                )
            excess_link = float(excess_link)
            if not np.isfinite(excess_link):
                raise ValueError(f"excess_link must be finite or None, got {excess_link}.")
            fractional = abs(excess_link - round(excess_link)) > TRACE_INTEGER_TOL
            if closed and link_reference == 'path' and fractional:
                raise ValueError(
                    f"For closed chains with link_reference='path' excess_link counts whole turns "
                    f"added to the twist-relaxed ring and has to be an integer (got "
                    f"{excess_link:g}). With link_reference='lk0' the linking number becomes "
                    f"round(Lk0 + excess_link)."
                )

        # FENE limit: the longest bond for which LAMMPS evaluates the FENE term regularly
        fene = cls._tracepoint_fene_limits(topology)
        fene_bound = None
        if auto_fene:
            max_fene = None
            if topology.has_fene and fene is None:
                raise ValueError(
                    'The topology uses FENE bonds, but their coefficients are not available. '
                    'Pass max_fene explicitly, or max_fene=None to build the configuration '
                    'without a limit.'
                )
            if fene is not None:
                rc, r0 = fene
                fene_bound = rc + (r0 - rc) * np.sqrt(1.0 - FENE_RLOGARG_MIN)
                model_bond = np.linalg.norm(topology.groundstate[:, 3:], axis=1).max()
                if model_bond > fene_bound:
                    raise ValueError(
                        f'The bonds of the groundstate reach {model_bond:.4g}, beyond '
                        f'{fene_bound:.4g}, the longest bond for which LAMMPS evaluates the FENE '
                        f'term of the topology regularly (Rc = {rc:g}, R0 = {r0:g}). These lengths '
                        f'are in simulation units (unit length {topology.unit_length:g} nm); set_fene '
                        f'takes Rc and R0 in nm. Pass max_fene=None to build the configuration anyway.'
                    )

        # the curve is built in conf_units, such that the lengths passed in and reported (apart
        # from the FENE messages) are in that unit; to_sim converts them to simulation units
        if conf_units == 'sim':
            to_sim = 1.0
        else:
            to_sim = CONF_UNITS_TO_NM[conf_units] / topology.unit_length
        # the twist of a step is defined up to full turns; reduced by the nearest whole number of
        # turns (principal values in [-pi, pi] stay as they are) it counts as in the linking number
        # of the beads
        twist = topology.groundstate[:, 2]
        twist = twist - 2 * np.pi * np.round(twist / (2 * np.pi))
        rise = topology.groundstate[:, 5]
        groundstate = np.column_stack([twist, rise / to_sim])
        options = dict(
            num_poses=topology.nbp,
            closed=closed,
            rescale=rescale,
            max_fene=max_fene,
            tangent_persistence=tangent_persistence,
            smoothing=smoothing,
            first_triad=first_triad,
            twist_correction=twist_correction,
            return_info=True,
        )
        issued = []

        def note(msg: str) -> None:
            issued.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

        # Relative to Lk0 the twist-relaxed ring is built first; its linking number sets the turns
        # to add. Its warnings are held back until it is known which ring is returned, and are
        # issued if anything fails before.
        lk0_reference = closed and excess_link is not None and link_reference == 'lk0'
        turns = 0.0 if excess_link is None or lk0_reference else excess_link
        if lk0_reference:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    poses, info = tracepoints_to_poses(
                        tracepoints, groundstate, excess_link=turns, **options
                    )
                    failure = None
                except Exception as e:
                    failure = e
            if failure is not None:
                for warning in caught:
                    note(str(warning.message))
                raise failure
            pending = list(info.warnings)
        else:
            poses, info = tracepoints_to_poses(
                tracepoints, groundstate, excess_link=turns, **options
            )
            issued.extend(info.warnings)
            pending = []
        poses[:, :3, 3] *= to_sim

        try:
            if not rescale:
                hint = cls._tracepoint_unit_hint(info.stretch, conf_units, topology.unit_length)
                low, high = LMP_TOPOL_GROUNDSTATE_MIN_FACTOR, LMP_TOPOL_GROUNDSTATE_MAX_FACTOR
                if not low <= info.stretch <= high:
                    check = f"Check the length unit of the tracepoints (conf_units='{conf_units}')."
                    raise ValueError(
                        f'The poses are spaced {info.stretch:.4g} times the intrinsic rise, '
                        f'implausibly far from the rise of the model (accepted without rescale: '
                        f'{low:g} to {high:g}). {hint or check} Set rescale=True to scale the '
                        f'curve to the intrinsic rise.'
                    )
                if hint is not None:
                    note(hint)

            if fene_bound is not None:
                lengths = np.linalg.norm(bond_vectors(poses[:, :3, 3], closed), axis=1)
                over = np.flatnonzero(lengths > fene_bound * (1.0 + TRACE_FENE_TOL))
                if len(over) > 0:
                    raise ValueError(
                        f'The poses are spaced {info.stretch:.4g} times the intrinsic rise, which '
                        f'stretches {len(over)} bond(s) to up to {lengths.max():.4g}, beyond '
                        f'{fene_bound:.4g}, the longest bond for which LAMMPS evaluates the FENE '
                        f'term of the topology regularly (Rc = {rc:g}, R0 = {r0:g}; simulation '
                        f'units; 0-based bond indices: {_listed(over)}). Set rescale=True to '
                        f'scale the curve to the intrinsic rise, or pass max_fene explicitly '
                        f'(largest stretch beyond the rise in {conf_units}; None removes the '
                        f'limit).'
                    )
            if fene is not None:
                for msg in check_fene(poses, *fene, closed=closed, abort_is_error=False):
                    note(msg)

            if lk0_reference:
                lk0 = twist.sum() / (2 * np.pi)
                lk = cls._tracepoint_linking_number(poses, twist, info.closure_twist)
                if abs(lk - round(lk)) > CONF_LK_INTEGER_TOL:
                    note(
                        f'The linking number of the twist-relaxed ring ({lk:.3f}) is not close to '
                        f'an integer: its steps bend strongly or twist by almost 180 degrees. The '
                        f'excess link may be off by a turn.'
                    )
                turns = round(lk0 + excess_link) - round(lk)
                if turns != 0:
                    # same positions, other twist: the warnings of the returned ring replace those
                    # of the twist-relaxed ring, apart from its closure
                    closure = 'Closing the relaxed ring'
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        poses, info = tracepoints_to_poses(
                            tracepoints, groundstate, excess_link=turns, **options
                        )
                    poses[:, :3, 3] *= to_sim
                    pending = [msg for msg in pending if msg.startswith(closure)] + [
                        msg for msg in info.warnings if not msg.startswith(closure)
                    ]
        except Exception:
            for msg in pending:
                note(msg)
            raise
        for msg in pending:
            note(msg)

        # twist added to every step (closure of the ring and excess link)
        if closed:
            added = (info.closure_twist + 2 * np.pi * turns) / topology.nbp
        else:
            added = info.excess_twist_per_step
        if abs(added) > CONF_MAX_ADDED_TWIST:
            msg = (
                f'The excess link adds {np.degrees(added):.1f} degrees of twist per step (limit '
                f'{np.degrees(CONF_MAX_ADDED_TWIST):.0f} degrees).'
            )
            if lk0_reference:
                msg += (
                    " Relative to Lk0 the twist also compensates the writhe of the path; "
                    "link_reference='path' counts the excess link relative to the twist-relaxed "
                    "ring for this path."
                )
            raise ValueError(msg)
        flipped = np.flatnonzero(np.abs(twist + added) > np.pi)
        if len(flipped) > 0:
            # with the intrinsic twist in [-pi, pi], all such steps lie on the side of the added
            # twist
            direction = 'less' if added > 0 else 'more'
            note(
                f'The twist of {len(flipped)} step(s) exceeds 180 degrees in magnitude (0-based '
                f'steps: {_listed(flipped)}). In the rotation-vector convention these steps appear '
                f'twisted the other way round: linking numbers evaluated from the bead frames by '
                f'the shortest rotation (e.g. PyLk triads2link) count up to {len(flipped)} turn(s) '
                f'{direction}, whereas the elastic model, which evaluates deformations relative to '
                f'the groundstate, sees the intended twist.'
            )

        info.warnings = issued
        conf = cls.from_poses(poses, topology, mass=mass)
        if return_info:
            return conf, info
        return conf

    @staticmethod
    def _tracepoint_fene_limits(topology: CGRBPTopology) -> tuple[float, float] | None:
        """
        FENE onset Rc and divergence R0 of the topology (simulation units), or None if it has no
        FENE bonds or their coefficients are not available.
        """
        if not topology.has_fene:
            return None
        rc, r0 = getattr(topology, 'fene_Rc', None), getattr(topology, 'fene_R0', None)
        if rc is not None and r0 is not None:
            return float(rc), float(r0)
        # topologies read from a database keep (k, Rc, R0) only in their bond types
        coeffs = [
            bondtype.extra for bondtype in getattr(topology, 'bondtypes', [])
            if bondtype.extra is not None and len(bondtype.extra) == 3
        ]
        if not coeffs:
            return None
        coeffs = np.array(coeffs, dtype=float)
        return float(coeffs[:, 1].min()), float(coeffs[:, 2].min())

    @staticmethod
    def _tracepoint_unit_hint(stretch: float, conf_units: str, unit_length: float) -> str | None:
        """Hint at a length unit in which the curve through the tracepoints matches the chain."""
        from .conf_import import CONF_UNITS, CONF_UNITS_TO_NM
        from .tracepoints import TRACE_WARN_STRETCH

        if abs(stretch - 1.0) <= TRACE_WARN_STRETCH:
            return None

        def to_nm(unit: str) -> float:
            return unit_length if unit == 'sim' else CONF_UNITS_TO_NM[unit]

        for unit in CONF_UNITS:
            matched = abs(stretch * to_nm(unit) / to_nm(conf_units) - 1.0) <= TRACE_WARN_STRETCH
            if unit != conf_units and matched:
                return (
                    f"If the tracepoints were given in {unit} (conf_units='{unit}'), the curve "
                    f"through them would match the length of the chain. Check the length unit of "
                    f"the tracepoints."
                )
        return None

    @staticmethod
    def _tracepoint_linking_number(
        poses: np.ndarray, twist: np.ndarray, closure_twist: float
    ) -> float:
        """
        Linking number of a twist-relaxed ring built by tracepoints_to_poses (no excess link), every
        step counted with its intended twist (intrinsic twist plus closure_twist / nbp).

        A planar ring (beads and tangents in one plane) has no writhe: its linking number is the
        total angle by which the frames turn about the tangents relative to parallel transport,
        each step taken on the branch of its intended twist. The sum of the twist itself would miss
        it, since a bent step turns by more than its twist (by up to about pi*bend^2/8). Otherwise
        the linking number is that of the ribbon along the first triad axes, evaluated with PyLk.
        """
        from .conf_checks import bond_vectors
        from .tracepoints import _parallel_transport, _wrap

        pos = poses[:, :3, 3]
        target = twist + closure_twist / len(twist)
        _, sv, vt = np.linalg.svd(pos - pos.mean(axis=0), full_matrices=False)
        # the tangents have to lie in the plane as well (three beads always do)
        tangents_in_plane = np.abs(poses[:, :3, 2] @ vt[2]).max() <= 1e-8
        if sv[2] <= 1e-10 * sv[0] and tangents_in_plane:
            x, y, z = poses[:, :3, 0], poses[:, :3, 1], poses[:, :3, 2]
            # the turning angle of a step is minus its transport angle relative to the frames
            transport, _ = _parallel_transport(z, x, y, closed=True)
            return float(np.sum(target + _wrap(-transport - target)) / (2 * np.pi))
        try:
            from ..evals.PyLk import pylk
        except ImportError as e:
            raise ImportError(
                "Counting the excess link of a closed topology relative to Lk0 "
                "(link_reference='lk0') requires the linking number of the ring and thus the PyLk "
                "submodule (cgrbptools/evals/PyLk). link_reference='path' and excess_link=None do "
                "not need it."
            ) from e
        # a ribbon far narrower than the distance between the beads
        width = 0.02 * np.median(np.linalg.norm(bond_vectors(pos, closed=True), axis=1))
        # Between consecutive beads the ribbon turns the shortest way, which miscounts steps
        # twisted by about 180 degrees or more (strongly bent ones already below 180 degrees). The
        # frames are therefore turned back about the tangents by the intended twist, leaving small
        # turns per step, and the whole turns taken out are added back.
        turns = round(target.sum() / (2 * np.pi))
        step = target - (target.sum() - 2 * np.pi * turns) / len(target)
        angle = np.r_[0.0, np.cumsum(step)[:-1]]
        cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
        x, y = poses[:, :3, 0], poses[:, :3, 1]
        triads = poses[:, :3, :3].copy()
        triads[:, :, 0] = cos * x - sin * y
        triads[:, :, 1] = sin * x + cos * y
        return float(pylk.triads2link(pos, triads, radius=width, closed=True)) + turns

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
        