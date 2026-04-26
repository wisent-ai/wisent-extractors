from setuptools import setup, find_packages

setup(
    name="wisent-extractors",
    version="0.1.4",
    author="Lukasz Bartoszcze and the Wisent Team",
    author_email="lukasz.bartoszcze@wisent.ai",
    description="Benchmark extractors for lm-eval-harness and HuggingFace tasks, used by the wisent package family",
    url="https://github.com/wisent-ai/wisent-extractors",
    packages=find_packages(include=["wisent", "wisent.*"]),
    python_requires=">=3.9",
    install_requires=[
        "wisent>=0.10.0",  # ContrastivePair, logger, errors, constants live in wisent-core for now
        "datasets>=2.0",
        "huggingface_hub>=0.20",
        "lm_eval>=0.4.0",
        "requests>=2.0",
        "sympy>=1.12",
        "latex2sympy2_extended>=1.0.0",
        "pyyaml",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
