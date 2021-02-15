from setuptools import setup, find_packages

setup(
    name="drl-smartcity",
    version="0.1.0",
    description="Deep Reinforcement Learning with Keras-RL and Mesa",
    author="eescriba",
    url="https://github.com/eescriba/mesa-keras-rl",
    package_dir={"": "src/"},
    packages=[""],
    install_requires=[
        "mesa",
        "jupyter",
        "numpy",
        "matplotlib",
        "keras-rl2",
        "gym",
        "h5py",
        "pillow",
        "seaborn",
    ],
)