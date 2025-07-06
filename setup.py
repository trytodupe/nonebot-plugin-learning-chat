from setuptools import setup, find_packages

setup(
    name="nonebot-plugin-learning-chat",
    version="0.4.1",
    packages=find_packages(),
    install_requires=[
        # "nonebot2[fastapi]>=2.0.0,<3.0.0",
        "nonebot-adapter-onebot>=2.1,<3.0.0",
        "nonebot-plugin-apscheduler>=0.5.0,<1.0.0",
        "tortoise-orm>=0.20.0",
        "jieba>=0.42.1,<1.0.0",
        "ruamel.yaml>=0.17.21,<1.0.0",
        # "amis-python @ git+https://github.com/trytodupe/amis-python.git@449569b1d1517f9a03011521d749f0c80781b7d4",
        # "python-jose>=3.3.0,<4.0.0",
        "nonebot-plugin-tortoise-orm>=0.1.1,<1.0.0",
    ],
)
