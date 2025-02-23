from setuptools import setup, find_packages

setup(
    name="srp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "libp2p-python",  # P2P网络支持
        "protobuf",       # 消息序列化
        "syft",           # 联邦学习支持
        "torch",          # 深度学习支持
        "cryptography",   # 加密支持
        "numpy",          # 数值计算
        "asyncio",        # 异步IO支持
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="丝路协议(Silk Road Protocol) - 去中心化AI系统通信协议",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="AI, protocol, P2P, federated-learning, encryption",
    url="https://github.com/yourusername/srp",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)