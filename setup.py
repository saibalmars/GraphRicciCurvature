import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphRicciCurvature",
    version="0.4.1",
    author="Chien-Chun Ni",
    author_email="saibalmars@gmail.com",
    description="Compute discrete Ricci curvatures and Ricci flow on NetworkX graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saibalmars/GraphRicciCurvature",
    install_requires=[
        "networkx",
        "numpy",
        "cython",
        "cvxpy",
        "pot",
        "networkit>=6.0"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
