
import setuptools

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dialogue-bot",
    version="0.0.1",
    author="Matthias Cetto",
    author_email="matthias.cetto@unisg.ch",
    description="A chatbot-framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/ds-unisg/qa-chatbot/dialogue-bot.git",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: TODO",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data = {
            '': ['*.yml'],
    },
    python_requires='>=3.5',
    install_requires=reqs
)