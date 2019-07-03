from setuptools import setup

setup(
        name='GraphRicciCurvature',
        version='0.1',
        description='Compute discrete Ricci curvatures and Ricci flow on graphs',
        author='Chien-Chun Ni',
        setup_requires=[
                'setuptools>=18.0',
        ],
        install_requires=[
                'cvxpy',
                'networkx',
                'numpy',
        ],
        extras_require={
                'faster_apsp': ['networkit'],
        },
        packages=['GraphRicciCurvature'],
        license='LICENSE.txt',
)
