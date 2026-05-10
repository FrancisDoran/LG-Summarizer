FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

WORKDIR /usr/local/app


RUN apt-get update && apt-get install -y \
    git build-essential autoconf automake autoconf-archive libtool pkg-config \
    python3-dev swig flex libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/opencog/link-grammar.git && \
    cd link-grammar && \
    ./autogen.sh || (cat autogen.err && false) && \
    ./configure \
      --prefix=/usr/local \
      --with-python-sys-prefix \
      PYTHON="$(which python)" && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

ENV PYTHONPATH=/usr/lib/python3.12/site-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV HF_HOME=/workspace/hf-cache
ENV TRANSFORMERS_CACHE=/workspace/hf-cache

COPY . /usr/local/app/LG-Summarizer

WORKDIR /usr/local/app/LG-Summarizer/build/python_deps

RUN python -m pip install --break-system-packages -r requirements.txt

WORKDIR /usr/local/app/LG-Summarizer

CMD ["python", "-m", "model.training.training"]
