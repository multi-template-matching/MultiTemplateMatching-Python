import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Multi-Template-Matching",
    version="1.1",
    author="Laurent Thomas",
    author_email="laurent132.thomas@laposte.net",
    description="Object-recognition in images using multiple templates",
    long_description=long_description,
    long_description_content_type="text/markdown",
	keywords="object-recognition object-localization",
    url="https://github.com/LauLauThom/MultiTemplateMatching-Python",
    packages=["MTM"],
	install_requires=[
          'numpy',
		  'opencv-python',
		  'scikit-image',
		  'scipy'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
		"Topic :: Scientific/Engineering :: Image Recognition"
    ],
)