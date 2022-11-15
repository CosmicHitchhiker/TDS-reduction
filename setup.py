import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tdsreduction",
    # PEP440
    version="0.1.0.dev2",
    author="Vsevolod Lander",
    author_email="sevalander@gmail.com",
    description="Reduce data obtained at Transient Double-beam Spectrograph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CosmicHitchhiker/tdsreduction",
    project_urls={
      "Bug Tracker": "https://github.com/CosmicHitchhiker/tdsreduction/issues",
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords='astronomy spectroscopy longslit reduction',
    packages=setuptools.find_packages(),
    scripts = ['bin/tdspipeline'],
    python_requires=">=3.6",
    install_requires=[
       'numpy>=1.13',
       'scipy>=1.0',
       'astropy',
       'matplotlib',
       'argparse',
       'sklearn',
       'tqdm',
       'lacosmic',
    ]
)
