from setuptools import setup, find_packages
import os
import sys
import glob
import platform
import inspect
import fnmatch

from distutils.util import change_root

from numpy.distutils.core import DistutilsSetupError, setup
from numpy.distutils.ccompiler import get_default_compiler
from numpy.distutils.misc_util import Configuration

version_file = os.path.join('gpar', 'version.py')

with open(version_file) as vfile:
	__version__ = vfile.read().strip()

SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
UTIL_PATH = os.path.join(SETUP_DIRECTORY, "gpar", "util")
sys.path.insert(0, UTIL_PATH)
from libname import _get_lib_name
sys.path.pop(0)
LOCAL_PATH = os.path.join(SETUP_DIRECTORY, "setup.py")

def find_packages():

	modules = []

	for dirpath, _, filenames in os.walk(os.path.join(SETUP_DIRECTORY, "gpar")):
		if "__init__.py" in filenames:
			modules.append(os.path.relpath(dirpath, SETUP_DIRECTORY))
	return [_i.replace(os.sep, ".") for _i in modules]

def configuration(parent_package="", top_path=None):
	"""
	Config function mainly used to compile C code
	"""
	config = Configuration("", parent_package, top_path)
	path = os.path.join("gpar", "src")
	# files = glob.glob(os.path.join(path, "*.c"))
	files = os.path.join(path, "array.c")

	config.add_extension(_get_lib_name('array',add_extension_suffix=False),files)
	add_data_files(config)

	return config
def add_data_files(config):

	EXCLUDE_WILDCARDS = ['*py', '*.pyc', '*pyo', '*.pdf', '.git*']
	EXCLUDE_DIRS = ['src', '__pycache__']
	common_prefix = SETUP_DIRECTORY + os.path.sep

	for root, dirs, files in os.walk(os.path.join(SETUP_DIRECTORY, 'gpar')):
		root = root.replace(common_prefix, '')
		for name in files:
			if any(fnmatch.fnmatch(name, w) for w in EXCLUDE_WILDCARDS):
				continue
			config.add_data_files(os.path.join(root, name))

		for folder in EXCLUDE_DIRS:
			if folder in dirs:
				dirs.remove(folder)
	FORCE_INCLUDE_DIRS = [
						os.path.join(SETUP_DIRECTORY, ' gpar')]
	for folder in FORCE_INCLUDE_DIRS:
		for root, _, files in os.walk(folder):
			for filename in files:
				config.add_data_files(os.path.relpath(os.path.join(root, filename),
					SETUP_DIRECTORY))


def setupPackage():
	setup(
		name = 'gpar',
		version = __version__,
		description = 'A package for performing array processing',
		url = 'https://github.com/gnpang/gPar',
		author = 'Guanning Pang',
		author_email = 'yuupgn@gmail.com',
		license = 'MIT',
		packages=find_packages(),
		classifiers = [
			'Development Status :: 1 - Alpha',
			'Intended Audience :: Geo-scientists',
			'Topic :: Seismic array processing',
			'License :: MIT License',
			'Programming Language :: Python :: 2.7 :: C'],
		keywords = 'seismology array processing',
		install_requires=['obspy >= 1.1.1', 'numpy', 'pandas >= 0.24.2',
						  'matplotlib'],
		include_package_data=True,
		zip_safe = False,
		ext_package='gpar.lib',
		configuration=configuration
		)

if __name__ == '__main__':
	setupPackage()
