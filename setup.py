import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

# Read version from MTM/version.py
exec(open('MTM/version.py').read())


setuptools.setup(
	name="Multi-Template-Matching",
	version=__version__,
	author="Laurent Thomas",
	author_email="laurent132.thomas@laposte.net",
	description="Object-recognition in images using multiple templates",
	long_description=long_description,
	long_description_content_type="text/markdown",
	keywords="object-recognition object-localization",
	url="https://github.com/multi-template-matching/MultiTemplateMatching-Python",
	packages=["MTM"],
	install_requires=[
		  'numpy',
		  'opencv-python-headless>=4.5.4',
		  'scikit-image',
		  'scipy',
		  'pandas'
	  ],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering :: Image Recognition",
		"Intended Audience :: Science/Research"		
	],
)