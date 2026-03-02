from setuptools import setup, find_packages

setup(
    name             = "maira-ml",
    version          = "0.1.0",
    author           = "MAIRA Contributors",
    description      = "ML Agentic Intelligence for Research Automation — instrumentation-free ML project analyzer",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/yourusername/maira-ml",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "pandas",
        "numpy",
        "rich",
        "pydantic",
        "click",
        "requests",
        "groq",
    ],
    extras_require = {
        "gemini":    ["google-genai"],
        "anthropic": ["anthropic"],
    },
    entry_points = {
        "console_scripts": [
            "maira=maira.cli:main",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)