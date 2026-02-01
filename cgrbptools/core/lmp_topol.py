from __future__ import annotations

import sys
import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
import hashlib
import numbers
from collections.abc import Iterable
from pathlib import Path

from collections.abc import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

from ..io.path_methods import create_relative_path
from .unit_conversion import RescaleUnits


LMP_RBP_DIMS = 6
CGRBP_DEFAULT_BOND_STYLE = 'rbp'
CGRBP_DEFAULT_ANGLE_STYLE = 'rbp'
CGRBP_DEFAULT_DEHIDRAL_STYLE = 'rbp'
CGRBP_FENE_BOND_STYLE = 'rbpfene'

LMP_TOPOL_ID_NUM_BP = 'number of rigid bodies'  
LMP_TOPOL_ID_NUM_BONDS = 'number of bonds'
LMP_TOPOL_ID_NUM_ANGLES = 'number of angles'
LMP_TOPOL_ID_NUM_DIHEDRALS = 'number of dihedrals'
LMP_TOPOL_ID_NUM_BOND_TYPES = 'number of bond types'
LMP_TOPOL_ID_NUM_ANGLE_TYPES = 'number of angle types'
LMP_TOPOL_ID_NUM_DIHEDRAL_TYPES = 'number of dihedral types'          
LMP_TOPOL_ID_COUP_RANGE = 'coupling range'
LMP_TOPOL_ID_BOND_STYLE = 'bond style'
LMP_TOPOL_ID_ANGLE_STYLE = 'angle style'
LMP_TOPOL_ID_DIHEDRAL_STYLE = 'dihedral style'
LMP_TOPOL_ID_SEQS_SET = 'seqs set'
LMP_TOPOL_ID_SEQS_CENTERED = 'seqs centered'
LMP_TOPOL_ID_CHARS_PER_ATOM = 'chars per atom'
LMP_TOPOL_ID_CLOSED = 'closed'   
LMP_TOPOL_UNIT_LENGTH = 'unit length'
LMP_TOPOL_UNIT_ENERGY = 'unit energy'
LMP_TOPOL_SUBTRACT_GS = 'subtract groundstate' 

# Groundstate validation bounds (in nm)
# Translational components should be within these bounds to catch unit errors
LMP_TOPOL_GROUNDSTATE_REF_LENGTH = 0.34  # Reference length (nm) for one base pair step
LMP_TOPOL_GROUNDSTATE_MIN_FACTOR = 0.33  # Minimum as fraction of reference
LMP_TOPOL_GROUNDSTATE_MAX_FACTOR = 3.00  # Maximum as fraction of reference

##################################################################################################################
##################################################################################################################
##################################################################################################################
# Auxiliary Methods

#########################################################
# Convert list of values to string
def number2str(vals: Iterable[float], decimals: int | None = None) -> str:
    if not isinstance(vals, Iterable):
        raise TypeError("vals must be an iterable of numbers")
    if decimals is not None:
        if not isinstance(decimals, int):
            raise TypeError("decimals must be an int or None")
        if decimals < 0:
            raise ValueError("decimals must be non-negative")
    out = []
    for val in vals:
        if not isinstance(val, numbers.Real):
            raise TypeError(f"vals must contain only real numbers, got {type(val)}")
        out.append(np.round(val, decimals=decimals) if decimals is not None else val)
    return " ".join(str(val) for val in out)

#########################################################
# Sparse handling

def to_dense(x):
    if x is None:
        return None
    if sp.sparse.issparse(x):
        return x.toarray()
    else:
        return np.asarray(x)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# Hashing and canonical key methods

def hash_bondcoeffs_128(
    vec: np.ndarray, 
    mat: np.ndarray,
    decimals: int | None = None,
    extra: np.ndarray | None = None,
    ) -> int:
    assert mat.shape == (LMP_RBP_DIMS,LMP_RBP_DIMS)
    assert vec.shape == (LMP_RBP_DIMS,)
    if decimals is not None:
        mat = np.round(mat, decimals=decimals)
        vec = np.round(vec, decimals=decimals)
        if extra is not None:
            extra = np.round(extra, decimals=decimals)
    # 128-bit digest (16 bytes)
    h = hashlib.blake2s(digest_size=16)
    h.update(mat.tobytes(order="C"))
    h.update(vec.tobytes(order="C"))
    if extra is not None: 
        extra = np.asarray(extra).ravel()
        h.update(extra.tobytes(order="C"))
    return int.from_bytes(h.digest(), byteorder="big")

def hash_anglecoeffs_128(
    vec1: np.ndarray,
    vec2: np.ndarray,
    mat: np.ndarray,
    decimals: int | None = None,
    extra: np.ndarray | None = None
    ) -> int:
    assert mat.shape == (LMP_RBP_DIMS,LMP_RBP_DIMS)
    assert vec1.shape == (LMP_RBP_DIMS,)
    assert vec2.shape == (LMP_RBP_DIMS,)
    if decimals is not None:
        mat = np.round(mat, decimals=decimals)
        vec1 = np.round(vec1, decimals=decimals)
        vec2 = np.round(vec2, decimals=decimals)
        if extra is not None:
            extra = np.round(extra, decimals=decimals)
    # 128-bit digest (16 bytes)
    h = hashlib.blake2s(digest_size=16)
    h.update(mat.tobytes(order="C"))
    h.update(vec1.tobytes(order="C"))
    h.update(vec2.tobytes(order="C"))
    if extra is not None: 
        extra = np.asarray(extra).ravel()
        h.update(extra.tobytes(order="C"))
    return int.from_bytes(h.digest(), byteorder="big")

def hash_dihedralcoeffs_128(
    vec1: np.ndarray,
    vec2: np.ndarray,
    mat: np.ndarray,
    decimals: int | None = None,
    extra: np.ndarray | None = None
    ) -> int:
    assert mat.shape == (LMP_RBP_DIMS,LMP_RBP_DIMS)
    assert vec1.shape == (LMP_RBP_DIMS,)
    assert vec2.shape == (LMP_RBP_DIMS,)
    if decimals is not None:
        mat = np.round(mat, decimals=decimals)
        vec1 = np.round(vec1, decimals=decimals)
        vec2 = np.round(vec2, decimals=decimals)
        if extra is not None:
            extra = np.round(extra, decimals=decimals)
    # 128-bit digest (16 bytes)
    h = hashlib.blake2s(digest_size=16)
    h.update(mat.tobytes(order="C"))
    h.update(vec1.tobytes(order="C"))
    h.update(vec2.tobytes(order="C"))
    if extra is not None: 
        extra = np.asarray(extra).ravel()
        h.update(extra.tobytes(order="C"))
    return int.from_bytes(h.digest(), byteorder="big")

def canonical_key(
    mat: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray | None = None,
    decimals: int | None = None,
    extra: np.ndarray | None = None,
) -> tuple:
    assert mat.shape == (LMP_RBP_DIMS, LMP_RBP_DIMS)
    assert vec1.shape == (LMP_RBP_DIMS,)
    if vec2 is not None:
        assert vec2.shape == (LMP_RBP_DIMS,)
    if extra is not None:
        extra = np.asarray(extra).ravel()

    if decimals is None:
        mat_key  = mat.view(np.int64).ravel()
        vec1_key = vec1.view(np.int64)
        parts = [mat_key, vec1_key]
        if vec2 is not None:
            vec2_key = vec2.view(np.int64)
            parts.append(vec2_key)
        if extra is not None:
            extra_key = extra.view(np.int64)
            parts.append(extra_key)
        out = []
        for p in parts:
            out.extend(p.tolist())
        return tuple(out)

    scale = 10 ** decimals
    q_mat  = np.rint(mat  * scale).astype(np.int64)
    q_vec1 = np.rint(vec1 * scale).astype(np.int64)

    parts = [q_mat.ravel(), q_vec1]
    if vec2 is not None:
        q_vec2 = np.rint(vec2 * scale).astype(np.int64)
        parts.append(q_vec2)
    if extra is not None:
        q_extra = np.rint(extra * scale).astype(np.int64)
        parts.append(q_extra)
        
    out = []
    for p in parts:
        out.extend(p.tolist())
    return tuple(out)


##################################################################################################################
##################################################################################################################
# Bonds, Angles and Dihedrals

class RBPCoeffsBase(ABC):
    type_count: int = 0
    instances: list["RBPCoeffsBase"] = []
    registry: dict[tuple, "RBPCoeffsBase"] = {}

    def __init__(
        self,
        gs1: np.ndarray,
        stiffness_matrix: np.ndarray | spmatrix,
        decimals: int | None,
        gs2: np.ndarray | None = None,
        additional_coeffs: np.ndarray | None = None,
    ):
        cls = type(self)
        cls.instances.append(self)
        cls.type_count = len(cls.instances)
        self.type_id = cls.type_count

        self.decimals = decimals
        self._deleted = False

        self.X0_1 = to_dense(gs1)
        self.stiff = to_dense(stiffness_matrix)
        self.X0_2 = to_dense(gs2)
        self.extra = to_dense(additional_coeffs)
        if self.extra is not None:
            self.extra = np.asarray(self.extra).ravel()

        if decimals is not None:
            self.X0_1 = np.round(self.X0_1, decimals=decimals)
            if self.X0_2 is not None:
                self.X0_2 = np.round(self.X0_2, decimals=decimals)
            self.stiff = np.round(self.stiff, decimals=decimals)
            if self.extra is not None:
                self.extra = np.round(self.extra, decimals=decimals)

        self.hash = self._compute_hash()
        cls.registry[self.canonical_key] = self

    @classmethod
    def create(
        cls,
        gs1: np.ndarray,
        stiffness_matrix: np.ndarray | spmatrix,
        decimals: int | None,
        gs2: np.ndarray | None = None,
        *,
        check_existing: bool = True,
        additional_coeffs: np.ndarray | None = None,
    ):
        tmp_X0_1 = to_dense(gs1)
        tmp_stiff = to_dense(stiffness_matrix)
        tmp_X0_2 = to_dense(gs2)
        tmp_extra = to_dense(additional_coeffs)
        if tmp_extra is not None:
            tmp_extra = np.asarray(tmp_extra).ravel()

        if decimals is not None:
            tmp_X0_1 = np.round(tmp_X0_1, decimals=decimals)
            if tmp_X0_2 is not None:
                tmp_X0_2 = np.round(tmp_X0_2, decimals=decimals)
            tmp_stiff = np.round(tmp_stiff, decimals=decimals)
            if tmp_extra is not None:
                tmp_extra = np.round(tmp_extra, decimals=decimals)
                
        key = canonical_key(
            tmp_stiff, tmp_X0_1, vec2=tmp_X0_2, decimals=decimals, extra=tmp_extra
        )
        if check_existing:
            existing = cls.registry.get(key)
            if existing is not None:
                return existing

        return cls(gs1, stiffness_matrix, decimals, gs2=gs2, additional_coeffs=additional_coeffs)

    @abstractmethod
    def _compute_hash(self) -> int:
        ...

    @cached_property
    def canonical_key(self) -> tuple:
        return canonical_key(
            self.stiff,
            self.X0_1,
            vec2=self.X0_2,
            decimals=self.decimals,
            extra=self.extra,
        )

    def delete(self):
        if self._deleted:
            return
        cls = type(self)
        key = self.canonical_key
        if self in cls.instances:
            cls.instances.remove(self)
        if key in cls.registry and cls.registry[key] is self:
            del cls.registry[key]
        for i, inst in enumerate(cls.instances, start=1):
            inst.type_id = i
        cls.type_count = len(cls.instances)
        self.X0_1 = None
        self.X0_2 = None
        self.stiff = None
        self.extra = None
        self.hash = None
        self._deleted = True

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.hash != other.hash:
            return False
        return self.canonical_key == other.canonical_key

    def __hash__(self) -> int:
        return int(self.hash)


class RBPBondCoeffs(RBPCoeffsBase):
    type_count: int = 0
    instances: list["RBPBondCoeffs"] = []
    registry: dict[tuple, "RBPBondCoeffs"] = {}

    @property
    def X0(self):
        return self.X0_1
    
    @classmethod
    def reset_registry(cls):
        """Reset the global registry, instances, and type count."""
        cls.type_count = 0
        cls.instances = []
        cls.registry = {}

    def _compute_hash(self) -> int:
        return hash_bondcoeffs_128(self.X0_1, self.stiff, decimals=self.decimals, extra=self.extra)

    @classmethod
    def from_string(cls, line: str, bond_style: str | None = None, decimals: int | None = None) -> "RBPBondCoeffs":
        """ Convert a database bond coeffs line to an RBPBondCoeffs instance."""
        tokens = line.split()
        idx = 1  # Skip type_id at position 0
        
        # Extract style from string if not provided
        if bond_style is None:
            bond_style = tokens[idx]
            idx += 1
        
        # Extract extra coefficients if FENE style
        extra = None
        if bond_style == CGRBP_FENE_BOND_STYLE:
            extra = np.array([float(tokens[idx]), float(tokens[idx+1]), float(tokens[idx+2])], dtype=np.float64)
            idx += 3
        
        # Extract groundstate vector (6 values)
        X0 = np.array([float(tokens[idx + i]) for i in range(LMP_RBP_DIMS)], dtype=np.float64)
        idx += LMP_RBP_DIMS
        
        # Extract upper triangular stiffness matrix (21 values) and reconstruct symmetric matrix
        stiff = np.zeros((LMP_RBP_DIMS, LMP_RBP_DIMS), dtype=np.float64)
        for i in range(LMP_RBP_DIMS):
            for j in range(i, LMP_RBP_DIMS):
                value = float(tokens[idx])
                stiff[i, j] = value
                if i != j:
                    stiff[j, i] = value  # Mirror to lower triangular
                idx += 1
        
        # Create instance directly (not through create()) to preserve file order
        return cls(X0, stiff, decimals, additional_coeffs=extra)

    def to_string(self, hybrid:bool=False):
        dstr = f'{self.type_id}'
        if hybrid:
            dstr += f' rbp'
            
        if self.extra is not None:
            for x in self.extra:
                dstr += f' {x}'
        for i in range(LMP_RBP_DIMS):
            dstr += f' {self.X0_1[i]}'
        for i in range(LMP_RBP_DIMS):
            for j in range(i, LMP_RBP_DIMS):
                dstr += f' {self.stiff[i, j]}'
        return dstr


class RBPAngleCoeffs(RBPCoeffsBase):
    type_count: int = 0
    instances: list["RBPAngleCoeffs"] = []
    registry: dict[tuple, "RBPAngleCoeffs"] = {}

    @classmethod
    def reset_registry(cls):
        """Reset the global registry, instances, and type count."""
        cls.type_count = 0
        cls.instances = []
        cls.registry = {}

    def _compute_hash(self) -> int:
        return hash_anglecoeffs_128(self.X0_1, self.X0_2, self.stiff, decimals=self.decimals, extra=self.extra)

    @classmethod
    def from_string(cls, line: str, angle_style: str | None = None, decimals: int | None = None) -> "RBPAngleCoeffs":
        """ Convert a database angle coeffs line to an RBPAngleCoeffs instance."""
        tokens = line.split()
        idx = 1  # Skip type_id at position 0
        
        # Extract style from string if not provided
        if angle_style is None:
            angle_style = tokens[idx]
            idx += 1
        
        # Extract extra coefficients if non-default style (currently angles have no extra coeffs)
        extra = None
        # Future: check for non-default angle styles that might have extra coefficients
        # if angle_style != CGRBP_DEFAULT_ANGLE_STYLE:
        #     extra = np.array([float(tokens[idx]), ...], dtype=np.float64)
        #     idx += num_extra
        
        # Extract first groundstate vector (6 values)
        X0_1 = np.array([float(tokens[idx + i]) for i in range(LMP_RBP_DIMS)], dtype=np.float64)
        idx += LMP_RBP_DIMS
        
        # Extract second groundstate vector (6 values)
        X0_2 = np.array([float(tokens[idx + i]) for i in range(LMP_RBP_DIMS)], dtype=np.float64)
        idx += LMP_RBP_DIMS
        
        # Extract full stiffness matrix (36 values for 6x6 matrix)
        stiff = np.zeros((LMP_RBP_DIMS, LMP_RBP_DIMS), dtype=np.float64)
        for i in range(LMP_RBP_DIMS):
            for j in range(LMP_RBP_DIMS):
                stiff[i, j] = float(tokens[idx])
                idx += 1
        
        # Create instance directly (not through create()) to preserve file order
        return cls(X0_1, stiff, decimals, gs2=X0_2, additional_coeffs=extra)

    def to_string(self, hybrid:bool=False):
        dstr = f'{self.type_id}'
        if hybrid:
            dstr += f' rbp'
            
        if self.extra is not None:
            for x in self.extra:
                dstr += f' {x}'
        for i in range(LMP_RBP_DIMS):
            dstr += f' {self.X0_1[i]}'
        for i in range(LMP_RBP_DIMS):
            dstr += f' {self.X0_2[i]}'
        for i in range(LMP_RBP_DIMS):
            for j in range(LMP_RBP_DIMS):
                dstr += f' {self.stiff[i, j]}'
        return dstr


class RBPDihedralCoeffs(RBPCoeffsBase):
    type_count: int = 0
    instances: list["RBPDihedralCoeffs"] = []
    registry: dict[tuple, "RBPDihedralCoeffs"] = {}

    @classmethod
    def reset_registry(cls):
        """Reset the global registry, instances, and type count."""
        cls.type_count = 0
        cls.instances = []
        cls.registry = {}

    def _compute_hash(self) -> int:
        return hash_dihedralcoeffs_128(self.X0_1, self.X0_2, self.stiff, decimals=self.decimals, extra=self.extra)

    @classmethod
    def from_string(cls, line: str, dihedral_style: str | None = None, decimals: int | None = None) -> "RBPDihedralCoeffs":
        """ Convert a database dihedral coeffs line to an RBPDihedralCoeffs instance."""
        tokens = line.split()
        idx = 1  # Skip type_id at position 0
        
        # Extract style from string if not provided
        if dihedral_style is None:
            dihedral_style = tokens[idx]
            idx += 1
        
        # Extract extra coefficients if non-default style (currently dihedrals have no extra coeffs)
        extra = None
        # Future: check for non-default dihedral styles that might have extra coefficients
        # if dihedral_style != CGRBP_DEFAULT_DEHIDRAL_STYLE:
        #     extra = np.array([float(tokens[idx]), ...], dtype=np.float64)
        #     idx += num_extra
        
        # Extract first groundstate vector (6 values)
        X0_1 = np.array([float(tokens[idx + i]) for i in range(LMP_RBP_DIMS)], dtype=np.float64)
        idx += LMP_RBP_DIMS
        
        # Extract second groundstate vector (6 values)
        X0_2 = np.array([float(tokens[idx + i]) for i in range(LMP_RBP_DIMS)], dtype=np.float64)
        idx += LMP_RBP_DIMS
        
        # Extract full stiffness matrix (36 values for 6x6 matrix)
        stiff = np.zeros((LMP_RBP_DIMS, LMP_RBP_DIMS), dtype=np.float64)
        for i in range(LMP_RBP_DIMS):
            for j in range(LMP_RBP_DIMS):
                stiff[i, j] = float(tokens[idx])
                idx += 1
        
        # Create instance directly (not through create()) to preserve file order
        return cls(X0_1, stiff, decimals, gs2=X0_2, additional_coeffs=extra)

    def to_string(self, hybrid:bool=False):
        dstr = f'{self.type_id}'
        if hybrid:
            dstr += f' rbp'
            
        if self.extra is not None:
            for x in self.extra:
                dstr += f' {x}'
        for i in range(LMP_RBP_DIMS):
            dstr += f' {self.X0_1[i]}'
        for i in range(LMP_RBP_DIMS):
            dstr += f' {self.X0_2[i]}'
        for i in range(LMP_RBP_DIMS):
            for j in range(LMP_RBP_DIMS):
                dstr += f' {self.stiff[i, j]}'
        return dstr


##################################################################################################################
##################################################################################################################
# Bonds, Angles and Dihedrals

@dataclass
class RBPBond:
    instances: ClassVar[list["RBPBond"]] = []
    count: ClassVar[int] = 0

    id1: int
    id2: int
    bondcoeffs: RBPBondCoeffs
    index: int = field(init=False)

    def __post_init__(self):
        cls = type(self)
        cls.instances.append(self)
        cls.count = len(cls.instances)
        self.index = cls.count

    @classmethod
    def reset_registry(cls):
        """Reset the global instances and count."""
        cls.count = 0
        cls.instances = []

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)
    
    @classmethod
    def from_string(cls, line: str, bondcoeffs_list: list[RBPBondCoeffs]) -> "RBPBond":
        """Parse a bond connectivity line and create an RBPBond instance."""
        tokens = line.split()
        # Format: index type_id id1 id2
        bond_type_id = int(tokens[1])
        id1 = int(tokens[2])
        id2 = int(tokens[3])
        bondcoeffs = bondcoeffs_list[bond_type_id - 1]  # type_id is 1-indexed
        return cls(id1, id2, bondcoeffs)
            
    def to_string(self, atom_shift: int = 0, bond_shift: int = 0, bond_type_shift: int = 0):
        return f'{self.index + bond_shift} {self.bondcoeffs.type_id + bond_type_shift} {self.id1 + atom_shift} {self.id2 + atom_shift}'

@dataclass
class RBPAngle:
    instances: ClassVar[list["RBPAngle"]] = []
    count: ClassVar[int] = 0

    id1: int
    id2: int
    id3: int
    anglecoeffs: RBPAngleCoeffs
    index: int = field(init=False)

    def __post_init__(self):
        cls = type(self)
        cls.instances.append(self)
        cls.count = len(cls.instances)
        self.index = cls.count

    @classmethod
    def reset_registry(cls):
        """Reset the global instances and count."""
        cls.count = 0
        cls.instances = []

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)
    
    @classmethod
    def from_string(cls, line: str, anglecoeffs_list: list[RBPAngleCoeffs]) -> "RBPAngle":
        """Parse an angle connectivity line and create an RBPAngle instance."""
        tokens = line.split()
        # Format: index type_id id1 id2 id3
        angle_type_id = int(tokens[1])
        id1 = int(tokens[2])
        id2 = int(tokens[3])
        id3 = int(tokens[4])
        anglecoeffs = anglecoeffs_list[angle_type_id - 1]  # type_id is 1-indexed
        return cls(id1, id2, id3, anglecoeffs)
            
    def to_string(self, atom_shift: int = 0, angle_shift: int = 0, angle_type_shift: int = 0):
        return f'{self.index + angle_shift} {self.anglecoeffs.type_id + angle_type_shift} {self.id1 + atom_shift} {self.id2 + atom_shift} {self.id3 + atom_shift}'


@dataclass
class RBPDihedral:
    instances: ClassVar[list["RBPDihedral"]] = []
    count: ClassVar[int] = 0

    id1: int
    id2: int
    id3: int
    id4: int
    dihedralcoeffs: RBPDihedralCoeffs
    index: int = field(init=False)

    def __post_init__(self):
        cls = type(self)
        cls.instances.append(self)
        cls.count = len(cls.instances)
        self.index = cls.count

    @classmethod
    def reset_registry(cls):
        """Reset the global instances and count."""
        cls.count = 0
        cls.instances = []

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)
    
    @classmethod
    def from_string(cls, line: str, dihedralcoeffs_list: list[RBPDihedralCoeffs]) -> "RBPDihedral":
        """Parse a dihedral connectivity line and create an RBPDihedral instance."""
        tokens = line.split()
        # Format: index type_id id1 id2 id3 id4
        dihedral_type_id = int(tokens[1])
        id1 = int(tokens[2])
        id2 = int(tokens[3])
        id3 = int(tokens[4])
        id4 = int(tokens[5])
        dihedralcoeffs = dihedralcoeffs_list[dihedral_type_id - 1]  # type_id is 1-indexed
        return cls(id1, id2, id3, id4, dihedralcoeffs)

    def to_string(self, atom_shift: int = 0, dihedral_shift: int = 0, dihedral_type_shift: int = 0):
        return f'{self.index + dihedral_shift} {self.dihedralcoeffs.type_id + dihedral_type_shift} {self.id1 + atom_shift} {self.id2 + atom_shift} {self.id3 + atom_shift} {self.id4 + atom_shift}'


##################################################################################################################
##################################################################################################################
# Build Molecule Topology


class CGRBPTopology:
    """Coarse-grained rigid base-pair (CGRBP) DNA topology representation for LAMMPS simulations.
    
    This class manages the complete molecular topology of coarse-grained rigid base pair models(RBP). 
    It handles the construction, storage, and serialization of bond, angle, and dihedral interactions 
    between base pairs, along with their coefficient definitions and optional sequence information.
    
    The topology is built from groundstate configurations and stiffness matrices that
    encode the mechanical properties of DNA. The class automatically generates bond
    interactions between adjacent base pairs and angle/dihedral interactions for
    longer-range couplings based on a specified coupling range.
    
    Key Features
    ------------
    - Automatic topology generation from groundstate vectors and stiffness matrices
    - Support for both open and closed (circular) chain topologies
    - Coefficient deduplication to minimize LAMMPS type definitions
    - Optional FENE (Finite Extensible Nonlinear Elastic) bond potentials
    - Per-atom sequence string storage for DNA sequence tracking
    - Database serialization for saving and loading complete topologies
    - LAMMPS-compatible output formatting with configurable index shifts
    
    Topology Structure
    ------------------
    The topology consists of three levels of interactions:
    
    1. **Bonds**: Connect adjacent base pairs (coupling range 0)
       - Store groundstate configuration and local stiffness block
       - Optional FENE coefficients for nonlinear elasticity
       
    2. **Angles**: Connect base pair triads (coupling range 1)
       - Encode nearest-neighbor coupling effects
       - Store two groundstate vectors and off-diagonal stiffness block
       
    3. **Dihedrals**: Connect base pair quartets (coupling range ≥ 2)
       - Encode longer-range coupling effects
       - Same structure as angles but for non-adjacent pairs
    
    Each interaction type has associated coefficient objects that store the mechanical
    parameters. Identical coefficients are automatically deduplicated to minimize the
    number of LAMMPS types.
    
    Workflow
    --------
    1. Initialize a CGRBPTopology instance with desired parameters
    2. Set mechanical parameters using set_params() with groundstate and stiffness matrix
    3. Optionally add sequence information using set_sequence() or set_atom_seqs()
    4. Optionally enable FENE bonds using set_fene()
    5. Export to LAMMPS format or save to database file
    
    Attributes
    ----------
    nbp : int
        Number of base pairs (atoms) in the topology.
    nbps : int
        Number of junctions (base pair steps) in the topology.
    coupling_range : int
        Maximum coupling range for angle/dihedral interactions.
    decimals : int or None
        Number of decimal places for coefficient rounding.
    closed : bool
        Whether the topology represents a closed (circular) chain.
    bond_style, angle_style, dihedral_style : str
        LAMMPS interaction styles for each interaction type.
    bonds, angles, dihedrals : list
        Lists of RBPBond, RBPAngle, RBPDihedral instances.
    bondtypes, angletypes, dihedraltypes : list
        Lists of unique coefficient objects (RBPBondCoeffs, etc.).
    groundstate : ndarray
        Groundstate configuration vectors for each junction.
    stiffness_matrix : ndarray or sparse matrix
        Stiffness matrix encoding mechanical coupling.
    couplings_set : bool
        Whether interactions have been initialized.
    seqs_set : bool
        Whether sequence information has been assigned.
    atom_seqs : list of str, optional
        Per-atom sequence strings if assigned.
    
    See Also
    --------
    RBPBondCoeffs, RBPAngleCoeffs, RBPDihedralCoeffs : Coefficient classes
    RBPBond, RBPAngle, RBPDihedral : Interaction classes
    
    Examples
    --------
    Create a simple open chain topology:
    
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> 
    >>> # Define groundstate and stiffness for 10 base pairs (9 junctions)
    >>> groundstate = np.zeros((9, 6))  # 9 junctions, 6 DOF each
    >>> stiffness_matrix = csr_matrix((54, 54))  # 9*6 x 9*6 stiffness matrix
    >>> 
    >>> # Create topology
    >>> topology = CGRBPTopology(coupling_range=2, decimals=6, closed=False)
    >>> topology.set_params(groundstate, stiffness_matrix)
    >>> 
    >>> # Add sequence information
    >>> topology.set_sequence('ATCGATCGAT', chars_per_atom=1)
    >>> 
    >>> # Save to database
    >>> topology.write_database('output.db')
    
    Load a topology from a database file:
    
    >>> topology = CGRBPTopology.read_database('output.db', decimals=6)
    >>> print(f"Loaded topology with {topology.nbp} base pairs")
    
    Enable FENE bonds for enhanced stability:
    
    >>> topology.set_fene(k=30.0, Rc=1.5, R0=2.0)
    
    Notes
    -----
    - The class uses global registries for coefficient deduplication. Use reset_registry()
      class methods on coefficient classes when creating multiple independent topologies.
    - Closed topologies require nbp == nbps (same number of atoms and junctions).
    - The stiffness matrix should be symmetric and positive semi-definite.
    - Coefficient rounding via the decimals parameter helps numerical stability and
      deduplication but may introduce small errors in mechanical properties.
    """
    
    def __init__(
        self,
        coupling_range: int = 2,
        decimals: int | None = None,
        *,
        check_existing_types: bool = True,
        closed: bool = False,
    ):
        """
        Initialize a CGRBP topology object.

        Parameters
        ----------
        coupling_range : int, optional
            Maximum coupling range. Must be non-negative.
        sequence : str or None, optional
            Optional global sequence string.
        decimals : int or None, optional
            Number of decimals used when formatting coefficients.
        check_existing_types : bool, optional
            Whether to reuse existing LAMMPS types if available.
        closed : bool, optional
            Whether the topology represents a closed chain.
        """
        
        # decimals
        if decimals is not None:
            if not isinstance(decimals, int):
                raise TypeError("decimals must be an int or None.")
            if decimals < 0:
                raise ValueError("decimals must be >= 0.")
        self.decimals = decimals

        # flags
        if not isinstance(check_existing_types, bool):
            raise TypeError("check_existing_types must be a bool.")
        self.check_existing = check_existing_types

        if not isinstance(closed, bool):
            raise TypeError("closed must be a bool.")
        self._closed = closed
        self.couplings_set = False
        self.seqs_set = False
        
        self.set_coupling_range(coupling_range)
        
        # internal state flags (recommended defaults)
        self.couplings_set = False
        
        self.bond_style = CGRBP_DEFAULT_BOND_STYLE
        self.angle_style = CGRBP_DEFAULT_ANGLE_STYLE
        self.dihedral_style = CGRBP_DEFAULT_DEHIDRAL_STYLE
        
        self.groundstate = None
        
        self.stiffness_matrix = None
        self.extra_bond = None
        self.extra_angle = None
        self.extra_dihedral = None
        
        self.unit_length = 1.0
        self.unit_energy = 1.0
        
        
    @property
    def closed(self) -> bool:
        """Whether the topology uses closed boundary conditions."""
        return self._closed
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in the topology."""
        if not hasattr(self, 'nbp') or self.nbp is None:
            return 0
        return self.nbp
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds in this topology."""
        if not hasattr(self, 'bonds') or self.bonds is None:
            return 0
        return len(self.bonds)
    
    @property
    def num_angles(self) -> int:
        """Number of angles in this topology."""
        if not hasattr(self, 'angles') or self.angles is None:
            return 0
        return len(self.angles)
    
    @property
    def num_dihedrals(self) -> int:
        """Number of dihedrals in this topology."""
        if not hasattr(self, 'dihedrals') or self.dihedrals is None:
            return 0
        return len(self.dihedrals)
    
    @property
    def num_bond_types(self) -> int:
        """Number of unique bond types in this topology."""
        if not hasattr(self, 'bondtypes') or self.bondtypes is None:
            return 0
        return len(self.bondtypes)
    
    @property
    def num_angle_types(self) -> int:
        """Number of unique angle types in this topology."""
        if not hasattr(self, 'angletypes') or self.angletypes is None:
            return 0
        return len(self.angletypes)
    
    @property
    def num_dihedral_types(self) -> int:
        """Number of unique dihedral types in this topology."""
        if not hasattr(self, 'dihedraltypes') or self.dihedraltypes is None:
            return 0
        return len(self.dihedraltypes)
    
    @property
    def num_bonds_string(self) -> str:
        """String representation of the number of bonds."""
        return f"{self.num_bonds} bonds"

    @property
    def num_angles_string(self) -> str:
        """String representation of the number of angles."""
        return f"{self.num_angles} angles"
    
    @property
    def num_dihedrals_string(self) -> str:
        """String representation of the number of dihedrals."""
        return f"{self.num_dihedrals} dihedrals"    
    
    @property
    def num_bond_types_string(self) -> str:
        """String representation of the number of bond types."""
        return f"{self.num_bond_types} bond types"
    
    @property
    def num_angle_types_string(self) -> str:
        """String representation of the number of angle types."""
        return f"{self.num_angle_types} angle types"    
    
    @property
    def num_dihedral_types_string(self) -> str:
        """String representation of the number of dihedral types."""
        return f"{self.num_dihedral_types} dihedral types"
    
    @property
    def has_fene(self) -> bool:
        """Whether FENE bond interaction is active."""
        return self.bond_style == CGRBP_FENE_BOND_STYLE
    
    @property
    def sequence(self) -> str | None:
        """Get the global sequence string if set."""
        if not self.seqs_set:
            return None
        if hasattr(self, 'atom_seqs') and self.atom_seqs is not None:
            return ''.join(self.atom_seqs)
        return None
    
    @property 
    def composite_size(self) -> int:
        """Numbers of base pairs per atom. This is the same as chars_per_atom."""
        return self.chars_per_atom
    
    def get_groundstate(self, length_rescaled: bool = True, energy_rescaled: bool = True) -> np.ndarray:
        """
        Get the groundstate configuration.
        
        The groundstate is stored internally in rescaled units. The original can be retrieved by setting
        length_rescaled and energy_rescaled to False.

        Parameters
        ----------
        length_rescaled : bool, optional
            If True (default), return groundstate in rescaled units.
            If False, return in native physical units (the original unit_length).
        energy_rescaled : bool, optional
            If True (default), return groundstate in rescaled units.
            If False, return in native physical units (the original unit_energy).
            Note: Groundstate components don't directly depend on energy, but this is
            kept for API consistency with get_stiffness_matrix().

        Returns
        -------
        np.ndarray
            Groundstate configuration array with shape (nbps, 6).
            First 3 components are rotational, last 3 are translational.
        """
        if self.groundstate is None:
            raise ValueError("Groundstate is not set.")
        
        gs = self.groundstate.copy()
        if length_rescaled and energy_rescaled:
            return gs

        unit_length = 1.0 if length_rescaled else self.unit_length
        unit_energy = 1.0 if energy_rescaled else self.unit_energy
        rescale = RescaleUnits(length_factor=unit_length, energy_factor=unit_energy)
        return rescale.rescale_groundstate(self.groundstate) 
        

    def get_stiffness_matrix(self, length_rescaled: bool = True, energy_rescaled: bool = True) -> np.ndarray:
        """
        Get the stiffness matrix.
        
        The stiffness matrix is stored internally in rescaled units. The original can be retrieved by setting
        length_rescaled and energy_rescaled to False.

        Parameters
        ----------
        length_rescaled : bool, optional
            If True (default), return stiffness in rescaled units.
            If False, return in native physical units (the original unit_length).
        energy_rescaled : bool, optional
            If True (default), return stiffness in rescaled units.
            If False, return in native physical units (the original unit_energy).

        Returns
        -------
        np.ndarray or scipy.sparse matrix
            Stiffness matrix with shape (6*nbps, 6*nbps).
            Units: [energy] / ([length]^2 for translational-translational blocks,
                              [length] for rotational-translational blocks,
                              dimensionless for rotational-rotational blocks)
                              
        """
        if self.stiffness_matrix is None:
            raise ValueError("Stiffness matrix is not set.")
        
        sm = self.stiffness_matrix.copy()
        if length_rescaled and energy_rescaled:
            return sm

        unit_length = 1.0 if length_rescaled else self.unit_length
        unit_energy = 1.0 if energy_rescaled else self.unit_energy
        rescale = RescaleUnits(length_factor=unit_length, energy_factor=unit_energy)
        return rescale.rescale_stiffness(self.stiffness_matrix)
    
    
    def set_unit_length(self, unit_length: float) -> None:
        """
        Set the unit length for the topology.

        Parameters
        ----------
        unit_length : float
            Unit length in nanometers. Must be positive.
            
        Raises
        ------
        TypeError
            If unit_length is not a float or int.
        ValueError
            If unit_length is not positive.
        """
        if not isinstance(unit_length, (float, int)):
            raise TypeError("unit_length must be a float or int.")
        if unit_length <= 0:
            raise ValueError("unit_length must be positive.")
        if not self.couplings_set:
            raise ValueError("Couplings must be set before changing unit length. stiffness matrix and groundstate must be passed in units of nm.")
        
        if self.couplings_set:
            rescale_factor = self.unit_length / unit_length
            if rescale_factor != 1.0:
                rescale = RescaleUnits(length_factor=rescale_factor)
                self.groundstate,self.stiffness_matrix = rescale.rescale_model(self.groundstate,self.stiffness_matrix)
                self._init_couplings()
        self.unit_length = unit_length
        
  
    def reset_unit_length(self) -> None:
        """
        Reset the unit length to 1.0 nm.
        """
        self.set_unit_length(1.0)
        
    
    def set_unit_energy(self, unit_energy: float) -> None:
        """
        Set the unit energy for the topology.

        Parameters
        ----------
        unit_energy : float
            Unit energy in kT. Must be positive.
            
        Raises
        ------
        TypeError
            If unit_energy is not a float or int.
        ValueError
            If unit_energy is not positive.
        """
        if not isinstance(unit_energy, (float, int)):
            raise TypeError("unit_energy must be a float or int.")
        if unit_energy <= 0:
            raise ValueError("unit_energy must be positive.")
        if not self.couplings_set:
            raise ValueError("Couplings must be set before changing unit energy. stiffness matrix must be passed in units of kT.")
        
        
        if self.couplings_set:
            rescale_factor = self.unit_energy / unit_energy
            if rescale_factor != 1.0:
                rescale = RescaleUnits(energy_factor=rescale_factor)
                self.groundstate,self.stiffness_matrix = rescale.rescale_model(self.groundstate,self.stiffness_matrix)
                self._init_couplings()
        self.unit_energy = unit_energy
        
    def reset_unit_energy(self) -> None:
        """
        Reset the unit energy to 1.0 kT.
        """
        self.set_unit_energy(1.0)
    
    def reset_rescaling(self) -> None:
        """
        Reset both unit length and unit energy to 1.0.
        """
        self.set_unit_length(1.0)
        self.set_unit_energy(1.0)
    
    @closed.setter
    def closed(self, value: bool) -> None:
        self.set_closed(value)


    def set_closed(self, closed: bool | None) -> None:
        """
        Set whether the topology represents a closed chain.

        If ``closed`` is None, this method does nothing. If the value changes,
        the number of atoms is updated and couplings are rebuilt (if already set).
        """
        if closed is None:
            return

        if not isinstance(closed, bool):
            raise TypeError("closed must be a bool or None.")

        if not hasattr(self, "_closed"):
            self._closed = closed
            return

        if self._closed == closed:
            return

        self._closed = closed
        if hasattr(self, "nbps") and self.nbps is not None:
            self.nbp = self.nbps if closed else self.nbps + 1

        if self.couplings_set:
            self._init_couplings()
    
    
    def set_coupling_range(self, coupling_range: int | None) -> None:
        """
        Set the maximum coupling range.

        The coupling range must be a non-negative integer and
        defines how many neighboring junctions are coupled when constructing
        bonds, angles, and dihedrals.
        """
        if coupling_range is None:
            return

        if not isinstance(coupling_range, int):
            raise TypeError("coupling_range must be an integer or None.")

        if coupling_range < 0:
            raise ValueError("coupling_range must be >= 0.")
        
        if hasattr(self, "coupling_range") and coupling_range == self.coupling_range:
            return

        self.coupling_range = coupling_range

        # Rebuild couplings if they already exist
        if self.couplings_set:
            self._init_couplings()
    
        
    def set_fene(
        self,
        k: float,
        Rc: float,
        R0: float
    ) -> None:
        """
        Set FENE bond parameters.

        The FENE (Finite Extensible Nonlinear Elastic) interaction provides a
        nonlinear elastic bond that diverges as the bond length approaches its
        maximum extension, preventing unphysical overstretching.

        Parameters
        ----------
        k : float
            FENE stiffness.
        Rc : float
            Distance at which the FENE interaction becomes active.
        R0 : float
            Maximum allowed bond extension.
        """
        
        if hasattr(self, "fene_k") and hasattr(self, "fene_Rc") and hasattr(self, "fene_R0") \
            and self.fene_k == k and self.fene_Rc == Rc and self.fene_R0 == R0:
                return
        
        self.fene_k = k
        self.fene_Rc = Rc
        self.fene_R0 = R0
        
        if self.fene_k <= 0:
            raise ValueError(f'Fene stiffness smaller or equal to zero (currently k = {k}).')
        if self.fene_R0 <= self.fene_Rc:
            raise ValueError(f'Fene Rc needs to be smaller or equal to R0 (currently: Rc = {Rc} R0 = {R0}).')
        
        self.bond_style = CGRBP_FENE_BOND_STYLE
        self.fene_coeffs = [self.fene_k,self.fene_Rc,self.fene_R0]
        self.extra_bond = np.array(self.fene_coeffs,dtype=np.float64)

        # Rebuild couplings if they already exist
        if self.couplings_set:
            self._init_couplings()
    
    
    def remove_fene(self) -> None:
        """
        Remove the FENE bond interaction and restore the default bond style.
        """
        if self.extra_bond is None:
            return
        
        self.fene_k = None
        self.fene_Rc = None
        self.fene_R0 = None
        self.bond_style = CGRBP_DEFAULT_BOND_STYLE
        self.fene_coeffs = None
        self.extra_bond = None
        # Rebuild couplings if they already exist
        if self.couplings_set:
            self._init_couplings()
        
        
    def set_params( self,
                    groundstate: np.ndarray, 
                    stiffness_matrix: np.ndarray | spmatrix,
                    coupling_range: int | None = None,
                    closed: bool | None = None,
                    extra_bond: np.ndarray | None = None,
                    extra_angle: np.ndarray | None = None,
                    extra_dihedral: np.ndarray | None = None,
                    validation: bool = False,
                  ) -> None:
        """ 
            Set interaction parameters (groundstate, stiffness and coupling ranges)
        """
        
        
        self.set_closed(closed)
        self.set_coupling_range(coupling_range)
        
        # Check groundstate consistency
        
        if groundstate.size == 0:
            raise ValueError("groundstate must be non-empty.")
        if len(groundstate.shape) == 1:
            if len(groundstate) % 6 != 0:
                raise ValueError('Invalid dimension of groundstate. Needs to be Nx6 (2-dimensional) or 6N (single dimension)')
            groundstate = groundstate.reshape((len(groundstate)//6,6))
        else:
            if len(groundstate[0]) != 6:
                raise ValueError('Second dimension of groundstate needs to contain 6 entries ')
        
        self.nbps = len(groundstate)
        self.nbp = self.nbps if self._closed else self.nbps+1
        
        
        ############# RECONSIDER THIS #########################
        # PROBLEM: this would require the composite size to be defined before calling set_params(), or as an argument
        # of set_params().
        #        
        # # Validate translational components of groundstate
        # # Check that the translational components (last 3 of each 6-vector) are in a reasonable range
        # # This helps catch cases where unit_length was not set to 1.0 nm
        # gs_array = np.asarray(groundstate)
        # trans_components = gs_array[:, 3:6]  # Last 3 components are translational
        
        # # Calculate magnitude of each translational vector
        # trans_magnitudes = np.linalg.norm(trans_components, axis=1)
        
        # min_bound = LMP_TOPOL_GROUNDSTATE_MIN_FACTOR * LMP_TOPOL_GROUNDSTATE_REF_LENGTH
        # max_bound = LMP_TOPOL_GROUNDSTATE_MAX_FACTOR * LMP_TOPOL_GROUNDSTATE_REF_LENGTH
        
        # invalid_indices = np.where((trans_magnitudes < min_bound) | (trans_magnitudes > max_bound))[0]
        
        # if len(invalid_indices) > 0:
        #     invalid_mags = trans_magnitudes[invalid_indices]
        #     err_msg = (
        #         f"Groundstate translational components are out of valid range [{min_bound:.3f}, {max_bound:.3f}] nm. "
        #         f"Found {len(invalid_indices)} invalid step(s):\n"
        #     )
        #     for idx, mag in zip(invalid_indices[:5], invalid_mags[:5]):  # Show first 5
        #         err_msg += f"  Step {idx}: magnitude = {mag:.6f} nm\n"
        #     if len(invalid_indices) > 5:
        #         err_msg += f"  ... and {len(invalid_indices) - 5} more\n"
        #     err_msg += (
        #         f"\nSuggestion: Check that groundstate and stiffness matrix are expressed in units of nm."
        #         f"Rescaling length units may be done after passing groundstate and stiffness matrix with set_unit_length(unit_length: float)."
        #     )
        #     raise ValueError(err_msg)
        
        # check stiffness matrix consistency
        if len(stiffness_matrix.shape) != 2:
            raise ValueError('Stiffness matrix must be two-dimensional.')
        if stiffness_matrix.shape[0] != self.nbps*6:
            raise ValueError('Dimension of stiffness matrix is incompatible with provided groundstate')
        if stiffness_matrix.shape != (self.nbps * 6, self.nbps * 6):
            raise ValueError("Stiffness matrix must be square with shape (6*nbps, 6*nbps).")
        
        self.groundstate = np.array(groundstate)
        self.stiffness_matrix = stiffness_matrix.copy()  
        
        if extra_bond is not None:
            self.extra_bond = extra_bond
        if extra_angle is not None:
            self.extra_angle = extra_angle
        if extra_dihedral is not None:
            self.extra_dihedral = extra_dihedral 
    
        self._init_couplings()
        
    
    def _reconstruct_groundstate(self) -> np.ndarray:
        """
        Reconstruct the groundstate array from stored bond coefficients.
        
        This method reverses the process in _init_couplings() by extracting
        the groundstate vectors from the bond coefficient objects. Each bond
        stores the groundstate configuration at its first junction position.
        
        Returns
        -------
        np.ndarray
            Reconstructed groundstate array of shape (nbps, 6) containing
            the configuration vector for each base pair step.
            
        Raises
        ------
        ValueError
            If couplings have not been set or no bonds exist.
        """
        if not self.couplings_set or not self.bonds:
            raise ValueError("Couplings must be set and bonds must exist to reconstruct groundstate.")
        
        # Initialize groundstate array
        groundstate = np.zeros((self.nbps, 6), dtype=np.float64)
        
        # Extract groundstate from bonds
        # Each bond stores the groundstate at its first junction (0-indexed: id1-1)
        for bond in self.bonds:
            junction_idx = bond.id1 - 1  # Convert 1-indexed atom ID to 0-indexed junction
            if junction_idx < self.nbps:
                groundstate[junction_idx] = bond.bondcoeffs.X0_1.copy()
        return groundstate

    def _reconstruct_stiffness_matrix(self) -> np.ndarray:
        """
        Reconstruct the full stiffness matrix from stored bond, angle, and dihedral coefficients.
        
        This method reverses the process in _init_couplings() by extracting
        stiffness blocks from coefficient objects and assembling them into
        the complete 6*nbps x 6*nbps stiffness matrix.
        
        The reconstruction places:
        - Bond stiffness blocks on diagonal blocks [i, i]
        - Angle stiffness blocks on off-diagonal blocks [i, i+1] and symmetric [i+1, i]
        - Dihedral stiffness blocks on off-diagonal blocks [i, i+k] and symmetric [i+k, i]
        
        Returns
        -------
        np.ndarray
            Reconstructed stiffness matrix of shape (6*nbps, 6*nbps).
            The matrix is symmetric.
            
        Raises
        ------
        ValueError
            If couplings have not been set.
        """
        if not self.couplings_set:
            raise ValueError("Couplings must be set to reconstruct stiffness matrix.")
        
        # Initialize stiffness matrix
        stiffmat = sp.sparse.lil_matrix((self.nbps * 6, self.nbps * 6), dtype=np.float64)
        # stiffmat = np.zeros((self.nbps * 6, self.nbps * 6), dtype=np.float64)
        
        # Place bond stiffness blocks (diagonal blocks [i, i])
        for bond in self.bonds:
            id1 = bond.id1 - 1  # Convert to 0-indexed
            
            # Bond stiffness is always on the diagonal block at the first junction
            i_start = id1 * 6
            i_end = (id1 + 1) * 6
            stiffmat[i_start:i_end, i_start:i_end] = bond.bondcoeffs.stiff.copy()
        
        # Place angle stiffness blocks (off-diagonal blocks)
        # Angles connect three consecutive atoms: angle(id1, id2, id3)
        # Stiffness is stored for coupling between id1 and id2
        for angle in self.angles:
            id1 = angle.id1 - 1  # Convert to 0-indexed
            id2 = angle.id2 - 1
            
            i_start = id1 * 6
            i_end = (id1 + 1) * 6
            j_start = id2 * 6
            j_end = (id2 + 1) * 6
            
            # Place stiffness block and its transpose (symmetric matrix)
            stiffmat[i_start:i_end, j_start:j_end] = angle.anglecoeffs.stiff.copy()
            stiffmat[j_start:j_end, i_start:i_end] = angle.anglecoeffs.stiff.T.copy()
        
        # Place dihedral stiffness blocks (off-diagonal blocks)
        # Dihedrals connect four atoms: dihedral(id1, id2, id3, id4)
        # Stiffness is stored for coupling between id1 and id3
        for dihedral in self.dihedrals:
            id1 = dihedral.id1 - 1  # Convert to 0-indexed
            id3 = dihedral.id3 - 1
            
            i_start = id1 * 6
            i_end = (id1 + 1) * 6
            j_start = id3 * 6
            j_end = (id3 + 1) * 6
            
            # Place stiffness block and its transpose (symmetric matrix)
            stiffmat[i_start:i_end, j_start:j_end] = dihedral.dihedralcoeffs.stiff.copy()
            stiffmat[j_start:j_end, i_start:i_end] = dihedral.dihedralcoeffs.stiff.T.copy()
        
        return stiffmat

    def _init_couplings(self) -> None:
        """
        Initialize bonds, angles, and dihedrals from groundstate and stiffness matrix.
        
        This private method constructs the complete topology by:
        1. Resetting all global registries to ensure clean state
        2. Creating bond interactions between adjacent atoms
        3. Creating angle interactions for nearest-neighbor couplings
        4. Creating dihedral interactions for longer-range couplings up to coupling_range
        5. Extracting unique coefficient types and storing copies in the topology
        
        The method handles both open and closed chain topologies, adjusting the
        connectivity pattern and atom indexing accordingly.
        
        This method should only be called internally by set_params(), set_closed(),
        set_coupling_range(), set_fene(), and remove_fene(). Users should not call
        this directly; use the public setter methods instead.
        
        Raises
        ------
        ValueError
            If closed topology has mismatched nbp and nbps values.
        """
        print('Initializing couplings...')
        
        # Reset all global registries to ensure clean state for this topology
        RBPBondCoeffs.reset_registry()
        RBPAngleCoeffs.reset_registry()
        RBPDihedralCoeffs.reset_registry()
        RBPBond.reset_registry()
        RBPAngle.reset_registry()
        RBPDihedral.reset_registry()
        
        # guardrail: this should never trigger!
        if self._closed and self.nbp != self.nbps:
            raise ValueError("Closed topology requires nbp == nbps (parameter arrays defined on junctions).")

        def _get_block(i: int, j: int):
            if self._closed:
                i = i % self.nbp
                j = j % self.nbp
            return to_dense(self.stiffness_matrix[i*6:(i+1)*6, j*6:(j+1)*6])

        bonds = []
        angles = []
        dihedrals = []
        
        # set bonds
        if not self._closed:
            for i in range(self.nbps):
                
                # bonds (local)
                id1 = i
                id2 = id1 + 1
                X0 = self.groundstate[id1]
                # REMOVE
                # M0 = to_dense(self.stiffness_matrix[id1*6:id2*6,id1*6:id2*6])
                M0 = _get_block(id1,id1)
                bonds.append(RBPBond(id1+1,id2+1,RBPBondCoeffs.create(X0,M0,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_bond)))
                
                id3 = i+2
                # angles (nearest neighbors)
                if self.coupling_range < 1 or id3 > self.nbps:
                    continue
                    
                X0_2 = self.groundstate[id2]
                # REMOVE
                # M1 = to_dense(self.stiffness_matrix[id1*6:id2*6,id2*6:id3*6])
                M1 = _get_block(id1,id2)
                angles.append(RBPAngle(id1+1,id2+1,id3+1,RBPAngleCoeffs.create(X0,M1,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_angle)))
                
                for k in range(2,self.coupling_range+1):
                    id3 = i+k
                    id4 = i+k+1
                    if id4 > self.nbps:
                        break
                    X0_2 = self.groundstate[id3]
                    # REMOVE
                    # Mk = to_dense(self.stiffness_matrix[id1*6:id2*6,id3*6:id4*6])
                    Mk = _get_block(id1,id3)
                    dihedrals.append(RBPDihedral(id1+1,id2+1,id3+1,id4+1,RBPDihedralCoeffs.create(X0,Mk,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_dihedral)))
        
        else:
            for i in range(self.nbp):
                
                # bonds (local)
                id1 = i
                id2 = (i+1)% self.nbp

                X0 = self.groundstate[i]
                # REMOVE
                # M0 = to_dense(self.stiffness_matrix[i*6:(i+1)*6,i*6:(i+1)*6])
                M0 = _get_block(i,i)
                bonds.append(RBPBond(id1+1,id2+1,RBPBondCoeffs.create(X0,M0,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_bond)))
                
                # angles (nearest neighbors)
                if self.coupling_range < 1:
                    continue
                    
                ii = i
                jj = (i+1) % self.nbp
                id1 = ii
                id2 = jj
                id3 = (jj+1) % self.nbp
                    
                X0_2 = self.groundstate[jj]
                # REMOVE
                # M1 = to_dense(self.stiffness_matrix[ii*6:(ii+1)*6,jj*6:(jj+1)*6])
                M1 = _get_block(ii,jj)
                angles.append(RBPAngle(id1+1,id2+1,id3+1,RBPAngleCoeffs.create(X0,M1,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_angle)))
                
                for k in range(2,self.coupling_range+1):
                    
                    jj = (i+k) % self.nbp
                    id3 = jj
                    id4 = (jj+1) % self.nbp
                    X0_2 = self.groundstate[jj]
                    # REMOVE
                    # Mk = to_dense(self.stiffness_matrix[ii*6:(ii+1)*6,jj*6:(jj+1)*6])
                    Mk = _get_block(ii,jj)
                    dihedrals.append(RBPDihedral(id1+1,id2+1,id3+1,id4+1,RBPDihedralCoeffs.create(X0,Mk,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_dihedral)))    
        
        # Store copies of instance lists, not references to global registries
        bondtypes = []
        if len(bonds) > 0:
            bondtypes = list(bonds[0].bondcoeffs.instances)
        
        angletypes = []    
        if len(angles) > 0:
            angletypes = list(angles[0].anglecoeffs.instances)
         
        dihedraltypes = []       
        if len(dihedrals) > 0:
            dihedraltypes = list(dihedrals[0].dihedralcoeffs.instances)
        
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        
        self.bondtypes      = bondtypes
        self.angletypes     = angletypes
        self.dihedraltypes  = dihedraltypes
        
        self.couplings_set = True
        
        
    def set_atom_seqs( self, seqs: Sequence[str], centered: bool = False) -> None:
        """
        Assign per-atom sequence strings.

        Each atom is associated with a string of characters. All atoms except
        the last must have the same number of characters; the last atom may
        have a different length. The number of provided sequences must match
        the number of atoms inferred from the coupling definition.

        Couplings must be set before calling this method.

        Parameters
        ----------
        seqs : Sequence of str
            Sequence strings assigned to each atom.
        centered : bool, optional
            Whether the sequences are considered centered. Center positions is chars_per_atom // 2.
            Default is False.
        """
        
        if not self.couplings_set:
            raise ValueError("Couplings need to be set before sequence assignment.")

        if isinstance(seqs, (str, bytes)) or not isinstance(seqs, Sequence):
            raise TypeError("seqs must be a sequence (e.g., list/tuple) of strings.")

        seqs = [seq.strip() for seq in seqs]

        if len(seqs) == 0:
            raise ValueError("seqs must be non-empty.")

        for i, s in enumerate(seqs):
            if not isinstance(s, str):
                raise TypeError(f"seqs[{i}] must be a str, got {type(s)}.")

        if len(seqs) != self.nbp:
            raise ValueError(
                f"Mismatch between number of atoms deduced from couplings (nbp={self.nbp}) "
                f"and provided number of seqs (len(seqs)={len(seqs)})."
            )
            
        self.chars_per_atom = len(seqs[0])
        for seq in seqs[:-1]:
            if len(seq) != self.chars_per_atom:
                raise ValueError(f'The same number of chars needs to be provided for all atoms, except for the last atom.')

        if self.closed and len(seqs[-1]) != self.chars_per_atom:
            raise ValueError(f'Closed topology requires the last atom to have the same number of chars as the rest of the chain')

        if len(seqs[-1]) > self.chars_per_atom:
            raise ValueError("The last atom sequence may be shorter, but not longer than the others.")
        
        center_pos = 0
        if centered:
            center_pos = self.chars_per_atom // 2
            if len(seqs[-1]) <= center_pos:
                min_seq_len = (self.nbp-1) * self.chars_per_atom + center_pos + 1
                raise ValueError(f'Sequence for atom {self.nbp} is too short to be centered. With {self.nbp} atoms the sequence has to contain at least {min_seq_len} characters ({len("".join(seqs))} character sequence provided).')

            if self.closed:
                err_msg = 'Centering is not supported for closed topologies. Consider shifting the sequence.'
                if len(''.join(seqs)) <= 200:
                    err_msg += f'\nCurrent sequence: {"".join(seqs)}'
                    err_msg += f'\nShifted sequence: {"".join(seqs)[center_pos:] + "".join(seqs)[:center_pos]}'
                raise ValueError(err_msg)

        self.seqs_centered = centered
        self.center_pos = center_pos
        self.atom_seqs = list(seqs)
        self.seqs_set = True
        
    
    def set_sequence(
        self, 
        seq: str, 
        chars_per_atom: int, 
        centered: bool = False,
        ) -> None:
        """
        Assign a global sequence and decompose it into per-atom blocks.

        The input sequence is split into consecutive blocks of size ``chars_per_atom``.
        All blocks have length ``chars_per_atom`` except possibly the last one, which may
        contain fewer characters if the sequence length is not an exact
        multiple of ``chars_per_atom``. The total number of blocks must match the number
        of atoms inferred from the coupling definition.

        Parameters
        ----------
        seq : str
            Global sequence string to be decomposed.
        chars_per_atom : int
            Number of characters per atom (coarse-graining level).
        centered : bool, optional
            Whether the resulting per-atom sequences are considered centered. Center positions is chars_per_atom // 2.
            Default is False.
        """
        if not self.couplings_set:
            raise ValueError("Couplings need to be set before sequence assignment.")

        if not isinstance(seq, str):
            raise TypeError("seq must be a string.")

        if not isinstance(chars_per_atom, int) or chars_per_atom <= 0:
            raise ValueError("chars_per_atom must be a positive integer.")

        n_blocks = (len(seq) + chars_per_atom - 1) // chars_per_atom
        if n_blocks != self.nbp:
            raise ValueError(
                f"Mismatch between sequence length and number of atoms: "
                f"ceil(len(seq)/chars_per_atom) = {n_blocks}, expected nbp = {self.nbp}."
            )

        # Decompose sequence into blocks of size chars_per_atom
        seqs = [seq[i:i + chars_per_atom] for i in range(0, len(seq), chars_per_atom)]
        self.set_atom_seqs(seqs, centered=centered)
    

    def connectivity_string(
        self,
        atom_shift: int = 0,
        bond_shift: int = 0,
        bond_type_shift: int = 0,
        angle_shift: int = 0,
        angle_type_shift: int = 0,
        dihedral_shift: int = 0,
        dihedral_type_shift: int = 0,
    ) -> str:
        """
        Format bonds, angles, and dihedrals sections as a string.
        
        This method generates LAMMPS-compatible connectivity sections containing
        bond, angle, and dihedral definitions. Each section lists interactions with
        their indices, type IDs, and participating atom IDs.
        
        The shift parameters enable combining multiple topologies by offsetting
        indices and type IDs to avoid conflicts. All shifts default to 0, producing
        output with original indices.
        
        Parameters
        ----------
        atom_shift : int, optional
            Offset added to all atom indices. Used when combining multiple molecules
            to ensure unique atom numbering. Default is 0.
        bond_shift : int, optional
            Offset added to bond interaction indices. Default is 0.
        bond_type_shift : int, optional
            Offset added to bond type IDs. Default is 0.
        angle_shift : int, optional
            Offset added to angle interaction indices. Default is 0.
        angle_type_shift : int, optional
            Offset added to angle type IDs. Default is 0.
        dihedral_shift : int, optional
            Offset added to dihedral interaction indices. Default is 0.
        dihedral_type_shift : int, optional
            Offset added to dihedral type IDs. Default is 0.
        
        Returns
        -------
        str
            Formatted connectivity sections containing Bonds, Angles, and Dihedrals
            sections as appropriate. Empty string if no interactions exist.
        """
        lines = []
        
        if len(self.bonds) > 0:
            lines.append('\nBonds\n\n')
            for bond in self.bonds:
                lines.append(f'{bond.to_string(atom_shift=atom_shift, bond_shift=bond_shift, bond_type_shift=bond_type_shift)}\n')
            lines.append('\n')
        
        if len(self.angles) > 0:
            lines.append('\nAngles\n\n')
            for angle in self.angles:
                lines.append(f'{angle.to_string(atom_shift=atom_shift, angle_shift=angle_shift, angle_type_shift=angle_type_shift)}\n')
            lines.append('\n')
        
        if len(self.dihedrals) > 0:
            lines.append('\nDihedrals\n\n')
            for dihedral in self.dihedrals:
                lines.append(f'{dihedral.to_string(atom_shift=atom_shift, dihedral_shift=dihedral_shift, dihedral_type_shift=dihedral_type_shift)}\n')
            lines.append('\n')
        
        return ''.join(lines)
    
    def sequence_string(self) -> str:
        """
        Format the per-atom sequences section as a string.

        This method generates a section containing the sequence strings
        assigned to each atom in the topology.

        Returns
        -------
        str
            Formatted Seqs section listing each atom's sequence string.
            Empty string if no sequences are set.
        """
        if not self.seqs_set:
            return ''
        
        lines = []
        lines.append('\nSeqs\n\n')
        for i, atom_seq in enumerate(self.atom_seqs):
            lines.append(f'{i+1} {atom_seq.upper()}\n')
        
        return ''.join(lines)
     
    def coeffs_string(self, hybrid: bool = False) -> str:
        """
        Format all coefficient sections as a single string.

        This method generates a combined string containing the bond,
        angle, and dihedral coefficient sections used in the topology.

        Returns
        -------
        str
            Formatted sections for Bond Coeffs, Angle Coeffs, and Dihedral Coeffs.
            Empty string if no coefficient types exist.
        """
        lines = []
        lines.append('\nBond Coeffs\n\n')
        for bondtype in self.bondtypes:
            lines.append(f'{bondtype.to_string(hybrid=hybrid)}\n')
        lines.append('\n')
        lines.append('\nAngle Coeffs\n\n')
        for angletype in self.angletypes:
            lines.append(f'{angletype.to_string(hybrid=hybrid)}\n')
        lines.append('\n')
        lines.append('\nDihedral Coeffs\n\n')
        for dihedraltype in self.dihedraltypes:
            lines.append(f'{dihedraltype.to_string(hybrid=hybrid)}\n')
        lines.append('\n')
        return ''.join(lines)
    
    @classmethod
    def read_database(
        cls,
        filename: Path| str,
        decimals: int | None = None,
        check_existing_types: bool = True,
        verbose: bool = False,
    ) -> "CGRBPTopology":
        """Read a database file and reconstruct a CGRBPTopology instance.
        
        This classmethod reads a database file created by write_database() and
        reconstructs the topology object with all metadata, coefficients, and
        optionally connectivity information. The unit_length and unit_energy
        scaling factors are restored from the file, ensuring that the coefficients
        are interpreted in the correct physical units.
        
        Parameters
        ----------
        filename : str or Path
            Path to the database file to read.
        decimals : int or None, optional
            Number of decimals for coefficient rounding. If None, uses value
            from file or no rounding.
        check_existing_types : bool, optional
            Whether to check for existing coefficient types when reconstructing.
            Default is True.
        verbose : bool, optional
            If True, print diagnostic information during parsing. Default is False.
        
        Returns
        -------
        CGRBPTopology
            Reconstructed topology object with all coefficients in rescaled units. 
            The unit_length and unit_energy attributes track the original physical 
            units for reference.
            
        Notes
        -----
        The coefficients stored in the database file are in rescaled units. The
        unit_length and unit_energy values in the file indicate what physical
        units these correspond to.
        """
        with open(filename, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        # Parse metadata
        metadata = {}
        i = 0
        while i < len(lines) and lines[i].strip():
            line = lines[i].strip()
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
            i += 1
        
        # Extract metadata values
        nbp = int(metadata.get(LMP_TOPOL_ID_NUM_BP, 0))
        coupling_range = int(metadata.get(LMP_TOPOL_ID_COUP_RANGE, 2))
        bond_style = metadata.get(LMP_TOPOL_ID_BOND_STYLE, CGRBP_DEFAULT_BOND_STYLE)
        angle_style = metadata.get(LMP_TOPOL_ID_ANGLE_STYLE, CGRBP_DEFAULT_ANGLE_STYLE)
        dihedral_style = metadata.get(LMP_TOPOL_ID_DIHEDRAL_STYLE, CGRBP_DEFAULT_DEHIDRAL_STYLE)
        seqs_set = bool(int(metadata.get(LMP_TOPOL_ID_SEQS_SET, 0)))
        seqs_centered = bool(int(metadata.get(LMP_TOPOL_ID_SEQS_CENTERED, 0))) if seqs_set else False
        chars_per_atom = int(metadata.get(LMP_TOPOL_ID_CHARS_PER_ATOM, 1)) if seqs_set else 1
        closed = bool(int(metadata.get(LMP_TOPOL_ID_CLOSED, 0)))
        unit_length = float(metadata.get(LMP_TOPOL_UNIT_LENGTH, 1.0))
        unit_energy = float(metadata.get(LMP_TOPOL_UNIT_ENERGY, 1.0))
                
        # Reset all registries before reading
        RBPBondCoeffs.reset_registry()
        RBPAngleCoeffs.reset_registry()
        RBPDihedralCoeffs.reset_registry()
        RBPBond.reset_registry()
        RBPAngle.reset_registry()
        RBPDihedral.reset_registry()
        
        # Parse sections
        sections = {}
        current_section = None
        section_lines = []
        
        for line in lines[i:]:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is a section header
            if line_stripped in ['Seqs', 'Bonds', 'Angles', 'Dihedrals', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs']:
                if current_section is not None:
                    sections[current_section] = section_lines
                current_section = line_stripped
                section_lines = []
            else:
                section_lines.append(line_stripped)

        # account for last section
        if current_section is not None:
            sections[current_section] = section_lines
        
        if verbose: 
            for key in sections:
                print(key)
        
        # Parse coefficient sections
        bondcoeffs_list = []
        if 'Bond Coeffs' in sections:
            for line in sections['Bond Coeffs']:
                if line:
                    bondcoeff = RBPBondCoeffs.from_string(line, bond_style, decimals)
                    if verbose: print(bondcoeff.to_string())

                    bondcoeffs_list.append(bondcoeff)
        
        if verbose: print('Parsed bond coeffs:', len(bondcoeffs_list))
        
        anglecoeffs_list = []
        if 'Angle Coeffs' in sections:
            for line in sections['Angle Coeffs']:
                if line:
                    anglecoeff = RBPAngleCoeffs.from_string(line, angle_style, decimals)
                    anglecoeffs_list.append(anglecoeff)
                    
        if verbose: print('Parsed angle coeffs:', len(anglecoeffs_list))
        
        dihedralcoeffs_list = []
        if 'Dihedral Coeffs' in sections:
            for line in sections['Dihedral Coeffs']:
                if line:
                    dihedralcoeff = RBPDihedralCoeffs.from_string(line, dihedral_style, decimals)
                    dihedralcoeffs_list.append(dihedralcoeff)
        
        if verbose: print('Parsed dihedral coeffs:', len(dihedralcoeffs_list))
        
        # Parse connectivity sections (if present)
        bonds = []
        if 'Bonds' in sections:
            for line in sections['Bonds']:
                if line:
                    bond = RBPBond.from_string(line, bondcoeffs_list)
                    bonds.append(bond)
        
        angles = []
        if 'Angles' in sections:
            for line in sections['Angles']:
                if line:
                    angle = RBPAngle.from_string(line, anglecoeffs_list)
                    angles.append(angle)
        
        dihedrals = []
        if 'Dihedrals' in sections:
            for line in sections['Dihedrals']:
                if line:
                    dihedral = RBPDihedral.from_string(line, dihedralcoeffs_list)
                    dihedrals.append(dihedral)
        
        # Parse sequences (if present)
        atom_seqs = []
        if 'Seqs' in sections:
            for line in sections['Seqs']:
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        atom_seqs.append(parts[1])
        
        # Create topology instance
        topology = cls(
            coupling_range=coupling_range,
            decimals=decimals,
            check_existing_types=check_existing_types,
            closed=closed,
        )
        
        # Set internal state
        topology.nbp = nbp
        topology.nbps = nbp if closed else nbp - 1
        topology.bond_style = bond_style
        topology.angle_style = angle_style
        topology.dihedral_style = dihedral_style
        
        # Set coefficients and interactions
        topology.bondtypes = bondcoeffs_list
        topology.angletypes = anglecoeffs_list
        topology.dihedraltypes = dihedralcoeffs_list
        topology.bonds = bonds
        topology.angles = angles
        topology.dihedrals = dihedrals
        topology.couplings_set = True
        
        topology.unit_length=unit_length
        topology.unit_energy=unit_energy
        
        topology.groundstate = topology._reconstruct_groundstate()
        topology.stiffness_matrix = topology._reconstruct_stiffness_matrix()
        
        # Set sequences if present
        if atom_seqs:
            topology.atom_seqs = atom_seqs
            topology.chars_per_atom = chars_per_atom
            topology.seqs_centered = seqs_centered
            topology.center_pos = chars_per_atom // 2 if seqs_centered else 0
            topology.seqs_set = True        
        return topology
    

    
    def write_database(
        self,
        filename: str | Path,
        add_extension: bool = True,
        include_connectivity: bool = True,
        ) -> None:
        """Write topology to a database file.
        
        This method serializes the complete topology including metadata, coefficient
        definitions, and optionally connectivity information to a database file. The
        file format is structured with labeled sections for easy parsing and can be
        read back using the read_database() classmethod for round-trip serialization.
        
        The database file contains:
        - Metadata section: nbp, coupling_range, styles, counts, sequence flags, closed,
          unit_length, unit_energy
        - Seqs section (optional): Per-atom sequence strings if sequences are set
        - Connectivity sections (optional): Bonds, Angles, Dihedrals with atom indices
        - Coefficient sections: Bond Coeffs, Angle Coeffs, Dihedral Coeffs (in LAMMPS units)
        
        Parameters
        ----------
        filename : str or Path
            Path where the database file will be written. Can be absolute or relative.
        add_extension : bool, optional
            If True, automatically adds or replaces the file extension with '.db'.
            If False, uses the filename exactly as provided. Default is True.
        include_connectivity : bool, optional
            If True, includes the connectivity sections (Bonds, Angles, Dihedrals) and
            their counts in the metadata. If False, only writes coefficient definitions
            and metadata. This is useful when only coefficient data is needed without
            the full topology structure. Default is True.
        
        Returns
        -------
        None
        
        Notes
        -----
        The coefficients are written in rescaled units. The unit_length and unit_energy 
        values in the metadata indicate what physical units these rescaled units correspond to.
        
        See Also
        --------
        read_database : Read a database file and reconstruct topology.
        """
        filename = Path(filename)
        
        if add_extension:
            filename = filename.with_suffix('.db')
        create_relative_path(filename)
        
        lines = []
        
        # metadata section

        def _format_metadata_lines(pairs):
            max_key_len = max(len(key) for key, _ in pairs)
            lines = []
            for key, value in pairs:
                pad = " " * (max_key_len - len(key) + 1)
                lines.append(f"{key}:{pad}{value}\n")
            return lines
        
        pairs = [
            (LMP_TOPOL_ID_NUM_BP, self.nbp),
            (LMP_TOPOL_ID_COUP_RANGE, self.coupling_range),
        ]

        if include_connectivity:
            pairs.extend([
                (LMP_TOPOL_ID_NUM_BONDS, self.num_bonds),
                (LMP_TOPOL_ID_NUM_ANGLES, self.num_angles),
                (LMP_TOPOL_ID_NUM_DIHEDRALS, self.num_dihedrals),
            ])

        pairs.extend([
            (LMP_TOPOL_ID_NUM_BOND_TYPES, self.num_bond_types),
            (LMP_TOPOL_ID_NUM_ANGLE_TYPES, self.num_angle_types),
            (LMP_TOPOL_ID_NUM_DIHEDRAL_TYPES, self.num_dihedral_types),
            (LMP_TOPOL_ID_BOND_STYLE, self.bond_style),
            (LMP_TOPOL_ID_ANGLE_STYLE, self.angle_style),
            (LMP_TOPOL_ID_DIHEDRAL_STYLE, self.dihedral_style),
            (LMP_TOPOL_SUBTRACT_GS, 0),
            (LMP_TOPOL_ID_SEQS_SET, int(self.seqs_set)),
        ])

        if self.seqs_set:
            pairs.extend([
                (LMP_TOPOL_ID_SEQS_CENTERED, int(self.seqs_centered)),
                (LMP_TOPOL_ID_CHARS_PER_ATOM, int(self.chars_per_atom)),
            ])

        pairs.extend([
            (LMP_TOPOL_ID_CLOSED, int(self.closed)),
            (LMP_TOPOL_UNIT_LENGTH, self.unit_length),
            (LMP_TOPOL_UNIT_ENERGY, self.unit_energy),
        ])
    
        lines = _format_metadata_lines(pairs)
        lines.append("\n")

        # sequences section
        if self.seqs_set:
            lines.append(self.sequence_string())
        
        # connectivity sections
        if include_connectivity:
            lines.append('\n')
            lines.append(self.connectivity_string())
        
        # coefficient sections
        lines.append(self.coeffs_string())

        with open(filename, 'w') as f:
            f.write(''.join(lines))
             