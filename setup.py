import setuptools
from pip.req import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphRicciCurvature",
    version="0.4.2",
    author="Chien-Chun Ni",
    author_email="saibalmars@gmail.com",
    description="Compute discrete Ricci curvatures and Ricci flow on NetworkX graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saibalmars/GraphRicciCurvature",
    install_requires=parse_requirements("requirements.txt"),
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
