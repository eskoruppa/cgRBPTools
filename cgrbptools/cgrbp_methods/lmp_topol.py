from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
import hashlib
from typing import Any, Callable, Dict, List, Tuple

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

LMP_RBP_DIMS = 6

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
    
    def __init__(self,
                 groundstate: np.ndarray, 
                 stiffmat: np.ndarray | spmatrix,
                 coupling_range: int = 2,
                 sequence: str = None,
                 decimals: int | None = None,
                 *,
                 check_existing_types: bool = True,
                 extra_bond: np.ndarray | None = None,
                 extra_angle: np.ndarray | None = None,
                 extra_dihedral: np.ndarray | None = None,
                 ):
        
        # Check groundstate consistency
        if len(groundstate.shape) == 1:
            if len(groundstate) % 6 != 0:
                raise ValueError('Invalid dimension of groundstate. Needs to be Nx6 (2-dimensional) or 6N (single dimension)')
            groundstate = groundstate.reshape((len(groundstate)//6,6))
        else:
            if len(groundstate[0]) != 6:
                raise ValueError('Second dimension of groundstate needs to contain 6 entries ')
        nbps = len(groundstate)
        
        # check stiffness matrix consistency
        if len(stiffmat.shape) != 2:
            raise ValueError('Stiffness matrix must be two-dimensional.')
        if stiffmat.shape[0] != nbps*6:
            raise ValueError('Dimension of stiffness matrix is incompatible with provided groundstate')
        
        
        stiffmat = stiffmat.toarray()
        
        self.groundstate = groundstate
        self.stiffmat = stiffmat       
        self.coupling_range = coupling_range
        self.sequence = sequence
        self.nbps = nbps
        self.decimals = decimals
        self.check_existing = check_existing_types
        
        self.extra_bond = extra_bond
        self.extra_angle = extra_angle
        self.extra_dihedral = extra_dihedral
        
        self.init_couplings()
        
    
    def init_couplings(self) -> None:
        
        bonds = []
        angles = []
        dihedrals = []
        
        # set bonds
        for i in range(self.nbps):
            
            # bonds (local)
            id1 = i
            id2 = i+1
            X0 = self.groundstate[i]
            M0 = self.stiffmat[id1*6:id2*6,id1*6:id2*6]
            bonds.append(RBPBond(id1+1,id2+1,RBPBondCoeffs.create(X0,M0,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_bond)))
            
            # angles (nearest neighbors)
            id3 = i+2
            if self.coupling_range < 1 or id3 > self.nbps:
                continue
                
            X0_2 = self.groundstate[id2]
            M1 = self.stiffmat[id1*6:id2*6,id2*6:id3*6]
            angles.append(RBPAngle(id1+1,id2+1,id3+1,RBPAngleCoeffs.create(X0,M1,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_angle)))
            
            for j in range(2,self.coupling_range+1):
                id3 = i+j
                id4 = i+j+1
                if id4 > self.nbps:
                    continue
                X0_2 = self.groundstate[id3]
                Mj = self.stiffmat[id1*6:id2*6,id3*6:id4*6]
                dihedrals.append(RBPDihedral(id1+1,id2+1,id3+1,id4+1,RBPDihedralCoeffs.create(X0,Mj,gs2=X0_2,decimals=self.decimals,check_existing=self.check_existing,additional_coeffs=self.extra_dihedral)))
        
        
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
