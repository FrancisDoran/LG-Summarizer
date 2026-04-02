# LG-Summarizer
This project aims to embed linguistic theory into neural networks via link grammar parse information.

## TOC
* [Google Colab Notebook](https://colab.research.google.com/github/FrancisDoran/LG-Summarizer/blob/main/LG.ipynb)
* **Build**
   * [Setup Link Grammar Parser on Linux](./build/lgp/setup_lgp.md)
   * [Setup Python Environment](./build/python_deps/setup_python.md)
       * Uses python for virtual environment, and pip for package management.
* **Demos**
   * [Baseline BART Summarizer](./baseline_model/model.py)
   * [Link Grammar Parser](./lg_parser_demo/main.py)
* **Our Model** 
   * [Custom Attention Mechanism](./model/LinkGramAttention.py)
   * [Token Transformer Layer](./model/TokenTransformerLayer.py)
