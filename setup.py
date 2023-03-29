# setup.py
import setuptools

setuptools.setup(
    name="pycaso",
    python_requires=">=3.8.10",
    version="1.0",
    author="LaMcube",
    author_email="eddy.caron@centralelille.fr",
    description="PYthon module for CAlibration of cameras by SOloff s method",
    long_description="PYthon module for CAlibration of cameras by SOloff s " +
    "method (PYCASO) provides an open-source Python-based framework for " + 
    "stereoscopic reconstructions from pairs of 2D images.",
    url='https://github.com/Eddidoune/Pycaso',
    project_urls={'Documentation':'No documentation for the moment'},
    packages=setuptools.find_packages(),
    ext_package='pycaso',
    license='CC BY-NC-ND 4.0 LICENSE',
    classifiers=['Development Status :: 4 - Beta ',
                 'Intended Audience :: Science/Research',
                 'Topic :: Software Development :: Build Tools',
                 'License :: CC BY-NC-ND 4.0 LICENSE',
                 'Programming Language :: Python :: 3.8.10',
                 'Natural Language :: English',
                 'Operating System :: OS Independent'],
    install_requires=["numpy>=1.23.1", 
                      "pandas>=1.3.2",
                      "matplotlib>=3.1.2",
                      "opencv-python>=4.5.3.56",
                      "sigfig>=1.3.1",
                      "scipy>=1.7.3",
                      "scikit-learn>=1.2.1",
                      "scikit-image>=0.18.3",
                      "seaborn>=0.11.2"],
    package_data={"": ["README.md","LICENSE"]})
