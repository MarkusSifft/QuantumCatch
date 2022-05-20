import os
from setuptools import setup, find_packages

import versioneer

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

#extra_files = package_files('strucscan/resources')


# with open('README.rst') as readme_file:
#  readme = readme_file.read()

setup(
name="strucscan",
version=versioneer.get_version(),
description="Light-weight python-based framework for high-throughput material simulation by AMS departement of ICAMS, Ruhr University Bochum",
url='https://git.noc.ruhr-uni-bochum.de/pietki8q/strucscan',
author='Isabel Pietka',
author_email='isabel.pietka@rub.de',
license='GPL3',

#  long_description=readme,

classifiers=['Development Status :: 5 - Production/Stable',
'Topic :: Scientific/Engineering :: Physics',
'License :: OSI Approved :: GNU GPL v3',
'Intended Audience :: Science/Research',
'Operating System :: OS Independent',
'Programming Language :: Python :: 3',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7'],

keywords='material properties',
#packages=find_packages(exclude=["strucscan"]),
#packages=find_packages('strucscan'),
packages=find_packages(include=['strucscan', 'strucscan.*']),
install_requires=['ase', 'numpy', 'matplotlib','plotly','pandas','numba','qutip','cachetools','tqdm','IPython',
'scipy', 'spglib', 'arrayfire','h5py'],
cmdclass=versioneer.get_cmdclass(),
include_package_data=True,
#  package_data={'': extra_files},
#  package_data={'': ["*.yaml", "*.sge", "*.slurm"]},
#  entry_points={
#  'console_scripts': [
#  'strucscan = strucscan.cli.main:main'
#  'strucscan = strucscan.cli:main'
#  ]
#  },
)