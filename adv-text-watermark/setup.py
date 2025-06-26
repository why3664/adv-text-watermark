# setup.py

from setuptools import setup, find_packages


# 一个常见的做法是从 requirements.txt 文件中读取依赖项
def parse_requirements(filename):
    """从给定的文件中加载行作为依赖项列表"""
    with open(filename, 'r') as f:
        lines = (line.strip() for line in f)
        return [line for line in lines if line and not line.startswith('#')]


setup(
    name="adv-text-watermark",
    version="0.1.0",
    author="<Your Name>",  # 您可以替换成您的名字
    author_email="<your.email@example.com>",  # 以及您的邮箱
    description="A project for adversarial text watermarking based on a three-layer attack framework.",
    long_description=open('README.md').read() if 'README.md' in __import__('os').listdir('.') else '',
    long_description_content_type="text/markdown",
    url="<URL to your project repository, e.g., on GitHub>",  # 如果有的话，可以放上项目链接

    # find_packages() 会自动查找项目中的所有包
    # 它会找到 adv_text_watermark 这个主包，以及它下面的所有子包 (data, models, attacks, etc.)
    packages=find_packages(
        where='.',
        include=['adv_text_watermark*']  # 明确指定包含的包
    ),

    # 声明项目的依赖项
    install_requires=parse_requirements('requirements.txt'),

    # 附加的元数据
    classifiers=[
        # 项目的成熟度
        "Development Status :: 3 - Alpha",

        # 项目的受众
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        # 项目的主题
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",

        # 许可证
        "License :: OSI Approved :: MIT License",  # 假设使用MIT许可证

        # 支持的Python版本
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    python_requires='>=3.8',  # 指定最低的Python版本要求
)
