from setuptools import setup, find_packages

setup(
    name="smart-cities-drl",
    version="0.1.0",
    description="Deep Reinforcement Learning for Smart Cities",
    author="eescriba",
    url="https://github.com/eescriba/smart-cities-drl",
    package_dir={"": "src/"},
    packages=[""],
    install_requires=[
        "mesa",
        "gym",
        "ray[rllib]",
    ],
)
