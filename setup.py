import setuptools

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphRicciCurvature",
    version="0.5.3.2",
    author="Chien-Chun Ni",
    author_email="saibalmars@gmail.com",
    description="Compute discrete Ricci curvatures and Ricci flow on NetworkX graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saibalmars/GraphRicciCurvature",
    setup_requires=["cython", "numpy"],  # to make sure these two are installed first for dependency.
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
