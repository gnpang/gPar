from __future__ import print_function, absolute_import
from __future__ import with_statement, nested_scopes, division, generators

from future.builtins import *
from future.utils import native_str

import ctypes as C 
import os
import numpy as np 

# from gpar.libname import _load_cdll

# clibarray = _load_cdll("array")
libdir = os.path.join(os.path.dirname(__file__),'lib')
libpath = os.path.join(libdir, 'libarray.so')
clibarray = C.CDLL(str(libpath))

clibarray.slidebeam.argtypes = [ C.c_int, C.c_int, C.c_int, 
				C.c_int, C.c_int, C.c_int, C.c_double,
				C.c_double, C.c_int, C.c_int,
				np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags=native_str('C_CONTIGUOUS')),
				np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
				np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags=native_str('C_CONTIGUOUS')),
				]
clibarray.slidebeam.restype = C.c_int

clibarray.slant.argtypes = [ C.c_int, C.c_int, C.c_int, 
				C.c_int, C.c_double,
				C.c_double, C.c_int, C.c_int,
				np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags=native_str('C_CONTIGUOUS')),
				np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
				np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags=native_str('C_CONTIGUOUS')),
				]
clibarray.slant.restype = C.c_int

clibarray.do_beam.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
							  np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags=native_str('C_CONTIGUOUS')),
							  C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_int, C.c_double, C.c_int, 
							  C.POINTER(C.c_double)]
clibarray.do_beam.restype = C.c_int

clibarray.do_beam_psw.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
							  np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags=native_str('C_CONTIGUOUS')),
							  C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_int, C.c_double, C.c_int, C.c_int,
							  C.POINTER(C.c_double)]
clibarray.do_beam_psw.restype = C.c_int

clibarray.do_beam_root.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
							  np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags=native_str('C_CONTIGUOUS')),
							  C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_int, C.c_double, C.c_int, C.c_int,
							  C.POINTER(C.c_double)]
clibarray.do_beam_root.restype = C.c_int

clibarray.get_mag.argtypes = [C.POINTER(C.c_double), C.c_int, C.c_int]
clibarray.get_mag.restype = C.c_int

clibarray.get_coherence.argtypes = [np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags=native_str('C_CONTIGUOUS')),
									np.ctypeslib.ndpointer(dtype=np.double, ndim=3, flags=native_str('C_CONTIGUOUS')),
									np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags=native_str('C_CONTIGUOUS')),
									C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
									C.c_int, C.c_double, C.c_int]
clibarray.get_coherence.restype = C.c_double

class C_COMPLEX(C.Structure):
	_fields_ = [("real", C.c_double),
				("imag", C.c_double)]


