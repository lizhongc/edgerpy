project('edgerpy', 'c', 'fortran',
  version : '0.1.0',
  license : 'GPL-2.0-or-later',
  meson_version : '>= 1.4.0',
  default_options : ['buildtype=plain', 'warning_level=0', 'b_ndebug=false'])

# Layout: c/ (all C), fortran/ (all Fortran), include/ (6 headers).
# Vendored files and their compile groups are listed in VENDORED.sha256, written by
# tools/vendor_sources.py. Compile flags reproduce the reference build (R 4.6.1, Debian 13):
#   R CMD config CFLAGS/FFLAGS for edgeR, limma and R's own nmath/appl/main code,
#   Debian lapack 3.12.1-6 FFLAGS (+ -frecursive, -O0 for dlamch.f) for BLAS/LAPACK.
# buildtype=plain keeps meson from adding its own -O/-g flags.
hardening = ['-g', '-fstack-protector-strong', '-fstack-clash-protection', '-fcf-protection']
r_cflags = ['-std=gnu23', '-O2', '-Wformat', '-Werror=format-security', '-Wdate-time', '-D_FORTIFY_SOURCE=2'] + hardening
r_fflags = ['-O2'] + hardening
lapack_fflags = ['-O2', '-frecursive'] + hardening
lapack_noopt_fflags = ['-O0', '-frecursive'] + hardening

# Linux and macOS. gcc (the reference) accepts every flag above, so there the flags are unchanged;
# clang and gfortran on macOS get the ones they support (no -fcf-protection on arm64, gnu2x before clang 18).
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')
# Only these flags vary between compilers; the others are kept as they are.
drop_c = []
foreach a : ['-std=gnu23', '-fstack-clash-protection', '-fcf-protection']
  if not cc.has_argument(a)
    drop_c += a
  endif
endforeach
drop_f = []
foreach a : ['-fstack-clash-protection', '-fcf-protection']
  if not fc.has_argument(a)
    drop_f += a
  endif
endforeach
flags = {'r_c' : r_cflags, 'r_f' : r_fflags, 'lapack_f' : lapack_fflags, 'lapack_noopt_f' : lapack_noopt_fflags}
kept = {}
foreach name, list : flags
  drop = name == 'r_c' ? drop_c : drop_f
  k = []
  foreach a : list
    if a not in drop
      k += a
    endif
  endforeach
  kept += {name : k}
endforeach
r_cflags = kept['r_c']
if '-std=gnu23' in drop_c
  r_cflags = ['-std=gnu2x'] + r_cflags
endif
r_fflags = kept['r_f']
lapack_fflags = kept['lapack_f']
lapack_noopt_fflags = kept['lapack_noopt_f']

# The platform part of R's configure results (see include/rapi.h): the size of long double.
# sinpi, cospi and tanpi always come from R's own c/nmath/cospi.c, never from the C library (glibc has
# them only from 2.41, macOS as __sinpi), so every platform runs the same code.
r_cflags += ['-DSIZEOF_LONG_DOUBLE=' + cc.sizeof('long double').to_string()]

openmp = dependency('openmp')
py = import('python').find_installation(pure : false)
numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check : true).stdout().strip()
inc = include_directories('include', 'c')

group_files = {}
foreach g : ['r-core-c', 'pkg-c', 'r-f', 'lapack-f', 'lapack-f-noopt']
  group_files += {g : run_command(py, 'tools/meson_sources.py', g, check : true).stdout().strip().split('\n')}
endforeach

# R 4.6.1 nmath and src/main excerpts, compiled as inside libR
r_core = static_library('r_core', group_files['r-core-c'],
  include_directories : inc, c_args : r_cflags + ['-DHAVE_CONFIG_H'], pic : true)
# edgeR kernels and limma weighted lowess (package code: -DNDEBUG, OpenMP)
kernels = static_library('kernels', group_files['pkg-c'],
  include_directories : inc, c_args : r_cflags + ['-DNDEBUG'], dependencies : openmp, pic : true)
# edgeR lowess/spline and R's LINPACK QR (R FFLAGS)
r_fortran = static_library('r_fortran', group_files['r-f'], fortran_args : r_fflags, pic : true)
# Netlib BLAS/LAPACK 3.12.1 closure (Debian FFLAGS; dlamch.f at -O0)
lapack = static_library('lapack', group_files['lapack-f'], fortran_args : lapack_fflags, pic : true)
lapack_noopt = static_library('lapack_noopt', group_files['lapack-f-noopt'], fortran_args : lapack_noopt_fflags, pic : true)
# edgerpy's R runtime shim
rshim = static_library('rshim', 'c/rshim.c', include_directories : inc, c_args : r_cflags, pic : true)

# Only PyInit__C is exported; every vendored symbol stays local to the module.
if host_machine.system() == 'darwin'
  exports = 'exports_macos.txt'
  export_args = ['-Wl,-exported_symbols_list,' + meson.current_source_dir() / exports]
else
  exports = 'exports.map'
  export_args = ['-Wl,--version-script=' + meson.current_source_dir() / exports]
endif
py.extension_module('_C',
  ['c/py_exports.c', 'c/rcompat.c'],
  include_directories : inc,
  c_args : r_cflags + ['-DNDEBUG', '-I' + numpy_inc],
  link_whole : [kernels, r_core, r_fortran, lapack, lapack_noopt, rshim],
  link_args : export_args + ['-lgfortran', '-lm'],
  link_depends : exports,
  dependencies : [openmp, py.dependency()],
  install : true,
  subdir : 'edgerpy')

install_subdir('src/edgerpy', install_dir : py.get_install_dir(), exclude_directories : ['__pycache__'])
