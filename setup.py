from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.3'
DESCRIPTION = 'Data factory for producing synthetic data.'
LONG_DESCRIPTION = 'Variantional Autoencoder used as a generative model for generating synthetic data.'

# Setting up
setup(
    name="aidatafactory",
    version=VERSION,
    author="Ngo Cuong",
    author_email="ngo.cuong.work@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'seaborn', 'tensorflow', \
        'tensorflow_probability', 'matplotlib', 'scipy'],
    keywords=['python', 'synthetic', 'data'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)