from setuptools import setup, find_packages

setup(
    name="ptolemy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "click>=8.1.7",
        "loguru>=0.7.2",
        "pydantic>=2.5.0",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "jinja2>=3.1.2",
        "pytest>=7.4.3",
    ],
    entry_points={
        "console_scripts": [
            "ptolemy=cli:cli",
        ],
    },
    author="PTOLEMY Team",
    author_email="info@ptolemy.ai",
    description="AI-Accelerated Development System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ptolemy/ptolemy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: ISC License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
