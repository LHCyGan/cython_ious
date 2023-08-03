from __future__ import print_function

from setuptools import Extension
from setuptools import setup
from distutils.command.build import build as _build
import os

# ref from https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class build(_build):
    def finalize_options(self):
        super().finalize_options()
        import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy as np
        # Obtain the numpy include directory.  This logic works across numpy versions.
        extension = next(m for m in self.distribution.ext_modules if m.name=='cython_ious')
        try:
            extension.include_dirs.append(np.get_include())
        except AttributeError:
            extension.include_dirs.append(np.get_numpy_include())

with open("README.md", "r") as fh:
    long_description = fh.read()

if os.name == 'nt':
    compile_args = {'gcc': ['/Qstd=c99']}
else:
    compile_args = ['-Wno-cpp']

ext_modules = [
    Extension(
        name='cython_ious',
        sources=['src/cython_ious.pyx'],
        extra_compile_args=compile_args,
    )
]

setup(
    name='cython_ious',
    setup_requires=["setuptools>=18.0", "Cython", "numpy"],
    install_requires=["Cython", "numpy"],
    ext_modules=ext_modules,
    cmdclass={'build': build},
    version='0.0.1',
    description='Standalone cython_ious',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='LHCyGan',
    author_email='autoencoder2000@163.com',
    url='https://github.com/LHCyGan/cython_ious.git',
    keywords=['cython_ious']
)