try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    import setuptools

setuptools.setup(
    name="safe-grid-gym",
    version="0.1",
    description="A gym interface for AI safety gridworlds created in pycolab.",
    long_description=(
        "Provides an OpenAI Gym interface for the AI safety gridworlds created "
        "by DeepMind. This allows to train reinforcement learning agents that "
        "use the OpenAI Gym interfece on the gridworld environments."
    ),
    url="https://github.com/david-lindner/safe-grid-gym/",
    author="David Lindner",
    author_email="dev@davidlindner.me",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai "
        "artificial intelligence "
        "gridworld "
        "gym "
        "rl "
        "reinforcement learning "
    ),
    install_requires=[
        "gym>=0.12",
        "ai-safety-gridworlds @ https://github.com/timorl/ai-safety-gridworlds/tarball/master#egg=ai-safety-gridworlds-1.5",
        "numpy",
        "pillow",
        "matplotlib",
    ],
    dependency_links=[
        "https://github.com/timorl/ai-safety-gridworlds/tarball/master#egg=ai-safety-gridworlds-1.5"
    ],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    test_suite="safe_grid_gym.tests",
    package_data={"safe_grid_gym.envs.common": ["*.ttf"]},
)
