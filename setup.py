from setuptools import setup, find_packages


# Function to read the contents of the requirements.txt file
def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="stream",
    version="0.1.0",
    packages=find_packages(exclude=["examples", "examples.*", "tests", "tests.*"]),
    install_requires=read_requirements(),
    include_package_data=True,
    description="A package for expanded topic modeling and metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="Anonymous",
    url="https://github.com/AnFreTh/STREAM",
    python_requires=">=3.6, <3.11",
)
