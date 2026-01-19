from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
import hashlib
import numbers
from typing import Any, Callable, Dict, List, Tuple, Iterable
from pathlib import Path

from collections.abc import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property


LMP_RBP_DIMS = 6
CGRBP_DEFAULT_BOND_STYLE = 'rbp'
CGRBP_DEFAULT_ANGLE_STYLE = 'rbp'
CGRBP_DEFAULT_DEHIDRAL_STYLE = 'rbp'
CGRBP_FENE_BOND_STYLE = 'rbpfene'

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
        stiffmat: np.ndarray | spmatrix,
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
        self.stiff = to_dense(stiffmat)
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
        stiffmat: np.ndarray | spmatrix,
        decimals: int | None,
        gs2: np.ndarray | None = None,
        *,
        check_existing: bool = True,
        additional_coeffs: np.ndarray | None = None,
    ):
        tmp_X0_1 = to_dense(gs1)
        tmp_stiff = to_dense(stiffmat)
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

        return cls(gs1, stiffmat, decimals, gs2=gs2, additional_coeffs=additional_coeffs)

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

    def _compute_hash(self) -> int:
        return hash_bondcoeffs_128(self.X0_1, self.stiff, decimals=self.decimals, extra=self.extra)

    @property
    def X0(self):
        return self.X0_1

    def to_str(self, hybrid:bool=False):
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

    def _compute_hash(self) -> int:
        return hash_anglecoeffs_128(self.X0_1, self.X0_2, self.stiff, decimals=self.decimals, extra=self.extra)

    def to_str(self, hybrid:bool=False):
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

    def _compute_hash(self) -> int:
        return hash_dihedralcoeffs_128(self.X0_1, self.X0_2, self.stiff, decimals=self.decimals, extra=self.extra)

    def to_str(self, hybrid:bool=False):
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

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)
            
    def to_str(self):
        return f'{self.index} {self.bondcoeffs.type_id} {self.id1} {self.id2}'

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

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)
            
    def to_str(self):
        return f'{self.index} {self.anglecoeffs.type_id} {self.id1} {self.id2} {self.id3}'


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

    def delete(self):
        cls = type(self)
        if self in cls.instances:
            cls.instances.remove(self)
            for i, inst in enumerate(cls.instances, start=1):
                inst.index = i
            cls.count = len(cls.instances)

    def to_str(self):
        return f'{self.index} {self.dihedralcoeffs.type_id} {self.id1} {self.id2} {self.id3} {self.id4}'


##################################################################################################################
##################################################################################################################
# Build Molecule Topology


class CGRBPTopology:
    
    def __init__(
        self,
        coupling_range: int = 2,
        decimals: int | None = None,
        *,
        check_existing_types: bool = False,
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
        
        self.set_coupling_range(coupling_range)
        
        # internal state flags (recommended defaults)
        self.couplings_set = False
        
        self.bond_style = CGRBP_DEFAULT_BOND_STYLE
        self.angle_style = CGRBP_DEFAULT_ANGLE_STYLE
        self.dihedral_style = CGRBP_DEFAULT_DEHIDRAL_STYLE
        
        self.groundstate = None
        
        self.stiffmat = None
        self.extra_bond = None
        self.extra_angle = None
        self.extra_dihedral = None
        
        self.couplings_set = False
        self.seqs_set = False
        
    
    @property
    def closed(self) -> bool:
        """Whether the topology uses closed boundary conditions."""
        return self._closed


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

        if getattr(self, "couplings_set", False):
            self.init_couplings()
    
    
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

        self.coupling_range = coupling_range

        # Rebuild couplings if they already exist
        if getattr(self, "couplings_set", False):
            self.init_couplings()
    
        
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
        
        self.fene_k = k
        self.fene_Rc = Rc
        self.fene_R0 = R0
        
        if self.fene_k <= 0:
            raise ValueError(f'Fene stiffness smaller or equal to zero (currently k = {k}).')
        if self.fene_R0 <= self.fene_Rc:
            raise ValueError(f'Fene Rc needs to be smaller or equal to R0 (currently: Rc = {Rc} R0 = {R0}).')
        
        self.bond_style = CGRBP_FENE_BOND_STYLE
        self.fene_coeffs = [self.fene_k,self.fene_Rc,self.fene_R0]
        self.extra_bond = number2str(self.fene_coeffs)
        
        # Rebuild couplings if they already exist
        if getattr(self, "couplings_set", False):
            self.init_couplings()
        
        
    def set_params( self,
                    groundstate: np.ndarray, 
                    stiffmat: np.ndarray | spmatrix,
                    coupling_range: int | None = None,
                    closed: bool | None = None,
                    extra_bond: np.ndarray | None = None,
                    extra_angle: np.ndarray | None = None,
                    extra_dihedral: np.ndarray | None = None,
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
        
        # check stiffness matrix consistency
        if len(stiffmat.shape) != 2:
            raise ValueError('Stiffness matrix must be two-dimensional.')
        if stiffmat.shape[0] != self.nbps*6:
            raise ValueError('Dimension of stiffness matrix is incompatible with provided groundstate')
        if stiffmat.shape != (self.nbps * 6, self.nbps * 6):
            raise ValueError("Stiffness matrix must be square with shape (6*nbps, 6*nbps).")
        
        self.groundstate = groundstate
        self.stiffmat = stiffmat  
        
        if extra_bond is not None:
            self.extra_bond = extra_bond
        if extra_angle is not None:
            self.extra_angle = extra_angle
        if extra_dihedral is not None:
            self.extra_dihedral = extra_dihedral 
        self.init_couplings()
        
    
    def init_couplings(self) -> None:
        """
            initialize bonds, angles and dihedrals from provided groundstate and stiffness matrix
        """
        
        # guardrail: this should never trigger!
        if self._closed and self.nbp != self.nbps:
            raise ValueError("Closed topology requires nbp == nbps (parameter arrays defined on junctions).")

        def _get_block(i: int, j: int):
            if self._closed:
                i = i % self.nbp
                j = j % self.nbp
            return to_dense(self.stiffmat[i*6:(i+1)*6, j*6:(j+1)*6])

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
                # M0 = to_dense(self.stiffmat[id1*6:id2*6,id1*6:id2*6])
                M0 = _get_block(id1,id1)
                bonds.append(RBPBond(id1+1,id2+1,RBPBondCoeffs.create(X0,M0,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_bond)))
                
                id3 = i+2
                # angles (nearest neighbors)
                if self.coupling_range < 1 or id3 > self.nbps:
                    continue
                    
                X0_2 = self.groundstate[id2]
                # REMOVE
                # M1 = to_dense(self.stiffmat[id1*6:id2*6,id2*6:id3*6])
                M1 = _get_block(id1,id2)
                angles.append(RBPAngle(id1+1,id2+1,id3+1,RBPAngleCoeffs.create(X0,M1,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_angle)))
                
                for k in range(2,self.coupling_range+1):
                    id3 = i+k
                    id4 = i+k+1
                    if id4 > self.nbps:
                        break
                    X0_2 = self.groundstate[id3]
                    # REMOVE
                    # Mk = to_dense(self.stiffmat[id1*6:id2*6,id3*6:id4*6])
                    Mk = _get_block(id1,id3)
                    dihedrals.append(RBPDihedral(id1+1,id2+1,id3+1,id4+1,RBPDihedralCoeffs.create(X0,Mk,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_dihedral)))
        
        else:
            for i in range(self.nbp):
                
                # bonds (local)
                id1 = i
                id2 = (i+1)% self.nbp

                X0 = self.groundstate[i]
                # REMOVE
                # M0 = to_dense(self.stiffmat[i*6:(i+1)*6,i*6:(i+1)*6])
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
                # M1 = to_dense(self.stiffmat[ii*6:(ii+1)*6,jj*6:(jj+1)*6])
                M1 = _get_block(ii,jj)
                angles.append(RBPAngle(id1+1,id2+1,id3+1,RBPAngleCoeffs.create(X0,M1,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_angle)))
                
                for k in range(2,self.coupling_range+1):
                    
                    jj = (i+k) % self.nbp
                    id3 = jj
                    id4 = (jj+1) % self.nbp
                    X0_2 = self.groundstate[jj]
                    # REMOVE
                    # Mk = to_dense(self.stiffmat[ii*6:(ii+1)*6,jj*6:(jj+1)*6])
                    Mk = _get_block(ii,jj)
                    dihedrals.append(RBPDihedral(id1+1,id2+1,id3+1,id4+1,RBPDihedralCoeffs.create(X0,Mk,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_dihedral)))    
        
        bondtypes = []
        if len(bonds) > 0:
            bondtypes = bonds[0].bondcoeffs.instances
        
        angletypes = []    
        if len(angles) > 0:
            angletypes = angles[0].anglecoeffs.instances
         
        dihedraltypes = []       
        if len(dihedrals) > 0:
            dihedraltypes = dihedrals[0].dihedralcoeffs.instances
        
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
            Whether the sequences are considered centered. Default is False.
        """
        
        if not self.couplings_set:
            raise ValueError("Couplings need to be set before sequence assignment.")

        if isinstance(seqs, (str, bytes)) or not isinstance(seqs, Sequence):
            raise TypeError("seqs must be a sequence (e.g., list/tuple) of strings.")

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
            
        if len(seqs[-1]) > self.chars_per_atom:
            raise ValueError("The last atom sequence may be shorter, but not longer than the others.")
            
        self.seqs_centered = centered
        self.atom_seqs = list(seqs)
        self.seqs_set = True
        
    
    def set_sequence(self, seq: str, cg: int, centered: bool = False) -> None:
        """
        Assign a global sequence and decompose it into per-atom blocks.

        The input sequence is split into consecutive blocks of size ``cg``.
        All blocks have length ``cg`` except possibly the last one, which may
        contain fewer characters if the sequence length is not an exact
        multiple of ``cg``. The total number of blocks must match the number
        of atoms inferred from the coupling definition.

        Parameters
        ----------
        seq : str
            Global sequence string to be decomposed.
        cg : int
            Number of characters per atom (coarse-graining level).
        centered : bool, optional
            Whether the resulting per-atom sequences are considered centered.
            Default is False.
        """
        if not self.couplings_set:
            raise ValueError("Couplings need to be set before sequence assignment.")

        if not isinstance(seq, str):
            raise TypeError("seq must be a string.")

        if not isinstance(cg, int) or cg <= 0:
            raise ValueError("cg must be a positive integer.")

        n_blocks = (len(seq) + cg - 1) // cg
        if n_blocks != self.nbp:
            raise ValueError(
                f"Mismatch between sequence length and number of atoms: "
                f"ceil(len(seq)/cg) = {n_blocks}, expected nbp = {self.nbp}."
            )

        # Decompose sequence into blocks of size cg
        seqs = [seq[i:i + cg] for i in range(0, len(seq), cg)]
        self.set_atom_seqs(seqs, centered=centered)
        
    
    def write_database(
        self,
        filename: str,
        add_extension: bool = True,
        ) -> None:
        
        if add_extension:
            filename = str(Path(filename).with_suffix('.db'))
        
        with open(filename,'w') as f:
            
            f.write(f'number of triads:         {self.nbp}\n')
            f.write(f'number of bonds types:    {len(self.bondtypes)}\n')
            f.write(f'number of angle types:    {len(self.angletypes)}\n')
            f.write(f'number of dihedral types: {len(self.dihedraltypes)}\n')
            f.write(f'bond style:               {self.bond_style}\n')
            f.write(f'angle style:              {self.angle_style}\n')
            f.write(f'dihedral style:           {self.dihedral_style}\n')
            f.write(f'seqs set:                 {int(self.seqs_set)}\n')
            if self.seqs_set:
                f.write(f'seqs centered:            {int(self.seqs_centered)}\n')
                f.write(f'chars per atom:           {int(self.chars_per_atom)}\n')
            f.write(f'closed:                   {int(self.seqs_centered)}\n')
            # f.write(f'scaling factor:           {SCALING FACTOR}\n')

            if self.seqs_set:
                f.write(f'\nSeqs\n\n')
                for i,atom_seq in enumerate(self.atom_seqs):
                    f.write(f'{i+1} {atom_seq.upper()}\n')
            
            if len(self.bondtypes) > 0:
                f.write(f'\nBond Coeffs\n\n')
                for bondtype in self.bondtypes:
                    f.write(f'{bondtype.to_str(hybrid=False)}\n')
                f.write('\n')

            if len(self.angletypes) > 0:
                f.write(f'\nAngle Coeffs\n\n')
                for angletype in self.angletypes:
                    f.write(f'{angletype.to_str(hybrid=False)}\n')
                f.write('\n')

            if len(self.dihedraltypes) > 0:
                f.write(f'\nDihedral Coeffs\n\n')
                for dihedraltype in self.dihedraltypes:
                    f.write(f'{dihedraltype.to_str(hybrid=False)}\n')
                f.write('\n')    
    

##################################################################################################################
##################################################################################################################
# Write Database file


# DEPRICATED!
def write_database(
    filename: str,
    topology: CGRBPTopology,
    add_extension: bool = True,
    seq: str | None = None,
    composite_size: int = 1,
    start_id: int | None = None,
    end_id: int | None = None 
    ) -> None:
    
    if add_extension:
        filename = str(Path(filename).with_suffix('.db'))
    
    with open(filename,'w') as f:
        
        f.write(f'{topology.nbps + 1} triads\n\n')
        
        if seq is not None:
            if end_id is not None:
                seq = seq[:end_id]
            if start_id is not None:
                seq = seq[start_id:]
            
            pseqs = [seq[ii*composite_size:(ii+1)*composite_size] for ii in range(topology.nbps+1)]
            pseqs[-1] = pseqs[-1][:1]
            
            f.write(f'\nSeqs\n\n')
            for i,pseq in enumerate(pseqs):
                f.write(f'{i+1} {pseq.upper()}\n')
        
        if len(topology.bondtypes) > 0:
            f.write(f'\nBond Coeffs\n\n')
            for bondtype in topology.bondtypes:
                f.write(f'{bondtype.to_str(hybrid=False)}\n')
            f.write('\n')

        if len(topology.angletypes) > 0:
            f.write(f'\nAngle Coeffs\n\n')
            for angletype in topology.angletypes:
                f.write(f'{angletype.to_str(hybrid=False)}\n')
            f.write('\n')

        if len(topology.dihedraltypes) > 0:
            f.write(f'\nDihedral Coeffs\n\n')
            for dihedraltype in topology.dihedraltypes:
                f.write(f'{dihedraltype.to_str(hybrid=False)}\n')
            f.write('\n')