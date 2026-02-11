from setuptools import setup, find_packages

setup(
    name="structdyn",
    version="0.3.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        'importlib-resources; python_version < "3.9"',
    ],
    extras_require={
        "test": ["pytest"],
    },
    description="A Python library for solving structural dynamics problems",
    author="Abinash Mandal",
    author_email="abinashmandal33486@gmail.com",
    url="https://github.com/learnstructure/structdyn",
)
