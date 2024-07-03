from setuptools import setup, find_packages

setup(
    name="WebRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "flask",
        "flask-socketio",
        #"python-dotenv",
        "safetensors",
        "Flask-Cors",
        "flax",
        #"gunicorn"
        #"gevent"
    ],
    author="Wilka Carvalho",
    author_email="wcarvalho92@gmail.com",
    description="A library to help create flask-based python web app for reinforcement learning experiments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcarvalho/webrl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
