from setuptools import setup, find_packages

setup(
    name="structdyn",
    version="0.2.0",
    packages=find_packages(),
    package_data={
        "structdyn": ["ground_motions/*.csv"],
    },
    install_requires=[
        "numpy",
        "matplotlib",
        'importlib-resources; python_version < "3.9"',
    ],
    description="A Python library for solving structural dynamics problems",
    author="Abinash Mandal",
    author_email="abinashmandal33486@gmail.com",
    url="https://github.com/learnstructure/structdyn",
)
