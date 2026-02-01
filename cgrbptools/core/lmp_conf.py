from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..SO3 import so3
from .lmp_topol import CGRBPTopology
from ..io.backups import backup_filename
from ..io.path_methods import create_relative_path
from ..PolyCG.polycg.out.visualization import visualize_chimerax, visualize_pdb, visualize_xyz
from .backmap import dna_backmap


LMP_RBP_DIMS = 6


@dataclass
class CGRBPConf:
    """
    DNA configuration for LAMMPS simulations.
    
    Stores the spatial configuration of coarse-grained rigid base pairs as
    SE3 transformation matrices (poses), along with mass information. Provides
    convenient access to positions, orientations (as triads or quaternions),
    and methods for generating LAMMPS input file strings.
    
    Attributes
    ----------
    poses : np.ndarray
        Array of SE3 transformation matrices with shape (nbp, 4, 4).
        Each 4x4 matrix represents the pose (position and orientation) of a base pair.
    mass : float
        Mass of each base pair.
    nbp : int
        Number of base pairs (set automatically from poses array length).
    topology : CGRBPTopology, optional
        Associated topology object (set via set_topology method).
    
    """
    poses: np.ndarray
    mass: float
    
    def __post_init__(self):
        self.nbp = len(self.poses)
    
    @property
    def positions(self):
        """np.ndarray: Positions of all base pairs with shape (nbp, 3)."""
        return self.poses[:,:3,3]

    @property
    def triads(self):
        """np.ndarray: Rotation matrices (triads) of all base pairs with shape (nbp, 3, 3)."""
        return self.poses[:,:3,:3]
    
    @property
    def quaternions(self):
        """np.ndarray: Unit quaternions representing orientations with shape (nbp, 4).
        
        Unit quaternions (normalized, with magnitude 1) are in [w, x, y, z] format,
        converted from rotation matrices.
        """
        return so3.mats2quats(self.poses[:,:3,:3])
    
    @property
    def bounds(self):
        """np.ndarray: Bounding box of all positions with shape (3, 2).
        
        Returns the minimum and maximum coordinates along each axis.
        Format: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        """
        return np.array([np.min(self.positions,axis=0),np.max(self.positions,axis=0)]).T
    
    def box_size(self, bounds:np.ndarray = None):
        """Calculate dimensions of the bounding box.
        
        Parameters
        ----------
        bounds : np.ndarray, optional
            Bounding box with shape (3, 2). If None, uses self.bounds.
        
        Returns
        -------
        np.ndarray
            Size along each axis [dx, dy, dz].
        """
        if bounds is None:
            bounds = self.bounds
        return bounds[:,1] - bounds[:,0]
    
    def extended_bounds(self, margin_fraction: float, square_box: bool = False):
        """Calculate extended bounding box with margins.
        
        Expands the bounding box by adding margins proportional to the current
        dimensions. Optionally creates a cubic box by extending smaller dimensions.
        
        Parameters
        ----------
        margin_fraction : float
            Fraction of the maximum dimension to use as margin on all sides.
            For example, 0.1 adds 10% of the max dimension as margin.
        square_box : bool, optional
            If True, extends the box to make it cubic (equal size in all dimensions).
            Default is False.
        
        Returns
        -------
        np.ndarray
            Extended bounding box with shape (3, 2).
        """
        bounds = self.bounds
        dimsize = self.box_size(bounds=bounds)
        margin = np.max(margin_fraction * dimsize)
        bounds[:,0] -= margin
        bounds[:,1] += margin
        if square_box:
            dim = self.box_size(bounds=bounds)
            mdim = np.max(dim)
            for i in range(len(bounds)):
                if dim[i] < mdim:
                    ext = (mdim-dim[i])*0.5
                    bounds[i,0] -= ext
                    bounds[i,1] += ext
        return bounds
    
    def metadata_section_string(self, topology: CGRBPTopology, box: np.ndarray, box_decimals: int = 1) -> None:
        """Generate LAMMPS metadata section string."""
        # Build content using list accumulation for better performance
        lines = []
        # Counts section
        lines.append(f'{len(self.positions)} atoms\n')
        lines.append(f'{len(topology.bonds)} bonds\n')
        lines.append(f'{len(topology.angles)} angles\n')
        lines.append(f'{len(topology.dihedrals)} dihedrals\n')
        lines.append(f'1 atom types\n')
        lines.append(f'{len(topology.bondtypes)} bond types\n')
        lines.append(f'{len(topology.angletypes)} angle types\n')
        lines.append(f'{len(topology.dihedraltypes)} dihedral types\n')
        lines.append(f'{self.nbp} ellipsoids\n')
        lines.append('\n')
        
        # Box dimensions
        rounded_box = np.round(box, decimals=box_decimals)
        lines.append(f'{rounded_box[0,0]} {rounded_box[0,1]} xlo xhi\n')
        lines.append(f'{rounded_box[1,0]} {rounded_box[1,1]} ylo yhi\n')
        lines.append(f'{rounded_box[2,0]} {rounded_box[2,1]} zlo zhi\n')
        
        # Masses section
        if len(topology.bondtypes) > 0:
            lines.append('\n\nMasses\n\n')
            for mass_string in self.mass_strings():
                lines.append(f'{mass_string}\n')
            lines.append('\n')
        return ''.join(lines)
            
    def atom_section_string(self):
        """Generate LAMMPS Atoms section string.
        
        Creates the full Atoms section for a LAMMPS data file.
        
        Returns
        -------
        str
            Complete Atoms section as a single string.
        """
        lines = []
        lines.append('\nAtoms\n\n')
        atom_strs = self.atom_strings()
        for atom_str in atom_strs:
            lines.append(f'{atom_str}\n')
        lines.append('\n')  
        return ''.join(lines)
    
    def atom_strings(self):
        """Generate LAMMPS atom data strings.
        
        Creates formatted strings for the Atoms section of a LAMMPS data file.
        Each string contains atom-ID, molecule-ID, position, and image flags.
        
        Returns
        -------
        list of str
            LAMMPS atom data strings, one per base pair.
        """
        strs = []
        for i,pos in enumerate(self.positions):
            pstr = f'{i+1} 1 {pos[0]} {pos[1]} {pos[2]} 1 1 1'
            strs.append(pstr)
        return(strs)
    
    def ellipsoid_section_string(self, elipsoid_shape: np.ndarray = np.array([1.0, 1.0000001, 0.9999999])):
        """Generate LAMMPS Ellipsoids section string.
        
        Creates the full Ellipsoids section for a LAMMPS data file.
        
        Parameters
        ----------
        elipsoid_shape : np.ndarray, optional
            Shape parameters for the ellipsoids. Default is [1.0, 1.0000001, 0.9999999].
        
        Returns
        -------
        str
            Complete Ellipsoids section as a single string.
        """
        lines = []
        lines.append('\nEllipsoids\n\n')
        ellipsoid_strs = self.ellipsoid_strings(elipsoid_shape=elipsoid_shape)
        for ellipsoid_str in ellipsoid_strs:
            lines.append(f'{ellipsoid_str}\n')
        lines.append('\n')
        return ''.join(lines)
    
    
    def ellipsoid_strings(self, elipsoid_shape: np.ndarray = np.array([1.0, 1.0000001, 0.9999999])):
        """Generate LAMMPS ellipsoid data strings.
        
        Creates formatted strings for the Ellipsoids section of a LAMMPS data file.
        Each string contains atom-ID, shape parameters, and unit quaternion orientation.
        
        Returns
        -------
        list of str
            LAMMPS ellipsoid data strings, one per base pair.
        """
        strs = []
        for i,quat in enumerate(self.quaternions):
            pstr = f'{i+1} {elipsoid_shape[0]} {elipsoid_shape[1]} {elipsoid_shape[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}'
            strs.append(pstr)
        return(strs)
    
    def mass_strings(self):
        """Generate LAMMPS mass data strings.
        
        Creates formatted strings for the Masses section of a LAMMPS data file.
        
        Returns
        -------
        list of str
            LAMMPS mass data strings (single entry for atom type 1).
        """
        return [f'{1} {self.mass}']
    
    def topology_set(self) -> bool:
        """Check if a topology is attached to this configuration.
        
        Returns
        -------
        bool
            True if a topology is set, False otherwise.
        """
        return hasattr(self, 'topology') and self.topology is not None
    
    def set_topology(self, topology: CGRBPTopology, overwrite: bool = False):
        """Attach a topology object to this configuration.
        
        Parameters
        ----------
        topology : CGRBPTopology
            Topology object to attach. Must have matching number of base pairs.
        
        Raises
        ------
        ValueError
            If topology.nbp does not match self.nbp.
        """
        if self.nbp != topology.nbp:
            raise ValueError(f"Number of base pairs in config ({self.nbp}) does not match topology ({topology.nbp})")
        
        if not overwrite and hasattr(self, 'topology') and self.topology is not None:
            if self.topology != topology:
                raise RuntimeError("Trying to set new topology with overwrite set to False.")
        self.topology = topology
        
    def poses_in_nm(self):
        """Return poses with positions in nanometers.
        
        If the current poses are in different units, this method converts
        the position components to nanometers.
        
        Returns
        -------
        np.ndarray
            Poses array with positions in nanometers.
        """
        if not hasattr(self, 'topology') or self.topology is None:
            raise RuntimeError("Topology must be set via set_topology() to determine unit length.")
        
        unit_length = self.topology.unit_length
        if unit_length == 1.0:
            return self.poses.copy()
        poses_nm = self.poses.copy()
        poses_nm[:,:3,3] *= unit_length
        return poses_nm
    
    def write_datafile(
        self,
        filename: str | Path,
        topology: CGRBPTopology,
        box: np.ndarray = None,
        hybrid: bool = False,
        include_coeffs: bool = False,
        add_extension: bool = True,
        box_decimals: int = 1,
        overwrite: bool = False
    ) -> Path:
        """Generate a LAMMPS data file for custom bond/angle/dihedral model.
        
        Parameters
        ----------
        filename : str or Path
            Output file path.
        topology : CGRBPTopology
            CGRBP topology containing bonds, angles, dihedrals.
        box : ndarray, optional
            Simulation box bounds with shape (3, 2), auto-calculated if None.
        hybrid : bool, optional
            Use hybrid style formatting for coefficients. Default is False.
        include_coeffs : bool, optional
            Include coefficient sections in output. Default is False.
        add_extension : bool, optional
            Automatically add .data extension if missing. Default is True.
        box_decimals : int, optional
            Decimal precision for box dimensions. Default is 1.
        overwrite : bool, optional
            If False (default), backup existing file before writing;
            if True, overwrite without backup.
        
        Returns
        -------
        Path
            Path object to the created data file.
        
        Raises
        ------
        TypeError
            If topology has incorrect type.
        ValueError
            If topology and config are inconsistent, or if positions are
            outside box bounds.
        OSError
            If file cannot be written.
        """
        # Input validation
        if not isinstance(topology, CGRBPTopology):
            raise TypeError(f"topology must be CGRBPTopology, got {type(topology).__name__}")
        if topology is None:
            raise ValueError("topology cannot be None")
        
        # Validate consistency
        if self.nbp != len(self.positions):
            raise ValueError(f"Inconsistent config: nbp={self.nbp} but positions has {len(self.positions)} entries")
        
        if self.nbp <= 0:
            raise ValueError("Configuration has no atoms to write")
        
        # Convert filename to Path and add extension
        filepath = Path(filename)
        if add_extension and filepath.suffix.lower() != '.data':
            filepath = filepath.with_suffix('.data')
        
        # Ensure directory exists
        create_relative_path(filepath)    
        
        # Calculate box if not provided
        if box is None:
            box = self.extended_bounds(margin_fraction=0.05, square_box=True)
        
        # Validate box bounds
        if box.shape != (3, 2):
            raise ValueError(f"box must have shape (3, 2), got {box.shape}")
        positions = self.positions
        if np.any(positions < box[:, 0]) or np.any(positions > box[:, 1]):
            raise ValueError("Some atom positions are outside the specified box bounds")
        
        lines = []
        
        # Header
        # Counts, box, masses sections
        lines.append('\n')
        lines.append('\n')
        lines.append(self.metadata_section_string(topology=topology, box=box, box_decimals=box_decimals))
        
        # Atoms section
        lines.append(self.atom_section_string())
        
        # Ellipsoids section
        lines.append(self.ellipsoid_section_string())
        
        # Coefficients sections (if requested)
        if include_coeffs:
            lines.append(topology.coeffs_string(hybrid=hybrid))
        
        # Connectivity sections     
        lines.append(topology.connectivity_string())
        
        # Backup existing file if needed
        if filepath.exists() and not overwrite:
            backup_path = backup_filename(filepath)
            try:
                filepath.rename(backup_path)
            except OSError as e:
                raise OSError(f"Failed to backup existing file to {backup_path}: {e}") from e
        
        # Write all content at once
        try:
            with open(filepath, 'w') as f:
                f.write(''.join(lines))
        
        except OSError as e:
            raise OSError(f"Failed to write data file to {filepath}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while generating data file: {e}") from e
        
        return filepath
    
    def visualize_chimerax(
        self, 
        base_fn: str | Path, 
        include_bps_triads: bool = True,
        include_beads: bool = True,
        bead_radius: float | None = None,
        ) -> None:
        
        """Visualize configuration in ChimeraX.
        
        Parameters
        ----------
        base_fn : str or Path
            Output PDB file path for ChimeraX visualization.
        bead_radius : float, optional
            Radius of beads for visualization. If None, no beads are drawn.
        include_beads : bool, optional
            Whether to include beads in the visualization. Default is True.
        """
        
        
        if self.topology is None:
            raise RuntimeError("Topology must be set via set_topology() before visualization.")
        
        if self.topology.composite_size > 1:
            bp_poses = dna_backmap(self,verbose=False)
        else:
            bp_poses = self.poses_in_nm()
        
        if include_beads:
            if bead_radius is None:
                if self.topology.composite_size > 1:
                    bead_radius = self.topology.composite_size*0.34*0.5
                else:
                    bead_radius = 0
        else:
            bead_radius = 0
          
        visualize_chimerax(
            base_fn, 
            self.topology.sequence, 
            self.topology.composite_size, 
            poses=bp_poses, 
            first_cg=self.topology.center_pos, 
            bead_radius=bead_radius,
            include_bps_triads=include_bps_triads
        ) 
        
    def visualize_pdb(
        self,
        base_fn: str | Path,   
    ) -> None:
        """Visualize configuration as PDB file.
        
        Parameters
        ----------
        base_fn : str or Path
            Output PDB file path.
        """
        if self.topology is None:
            raise RuntimeError("Topology must be set via set_topology() before visualization.")

        if self.topology.composite_size > 1:
            bp_poses = dna_backmap(self,verbose=False)
        else:
            bead_radius = 0
            bp_poses = self.poses_in_nm()
        visualize_pdb(
            base_fn, 
            self.topology.sequence, 
            poses=bp_poses
        )
        
    def visualize_xyz(
        self,
        base_fn: str | Path,   
    ) -> None:
        """Visualize configuration as XYZ file.
        
        Parameters
        ----------
        base_fn : str or Path
            Output XYZ file path.
        """

        visualize_xyz(
            base_fn, 
            1, 
            poses=self.poses,
        )
