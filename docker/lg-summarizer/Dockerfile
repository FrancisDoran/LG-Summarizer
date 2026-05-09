FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /usr/local/app


RUN apt-get update && apt-get install -y \
    git build-essential autoconf automake autoconf-archive libtool pkg-config python3-link-grammar \
    python3-dev swig python3-brlapi brltty \
    && rm -rf /var/lib/apt/lists/*


RUN git clone -b docker https://github.com/FrancisDoran/LG-Summarizer.git && \
	cd LG-Summarizer/build/python_deps && \
	./install_deps.sh

WORKDIR /usr/local/app/LG-Summarizer
