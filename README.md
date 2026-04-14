# LG-Summarizer
This project aims to embed linguistic theory into neural networks via link grammar parse information.

## TOC
* [Quick Usage](#test-the-model)
* [Google Colab Notebook](https://colab.research.google.com/github/FrancisDoran/LG-Summarizer/blob/main/LG.ipynb)
* **Build**
   * [Setup Link Grammar Parser on Linux](./build/lgp/setup_lgp.md)
   * [Setup Python Environment](./build/python_deps/setup_python.md)
       * Uses python for virtual environment, and pip for package management.
* **Demos**
   * [Baseline BART Summarizer](./baseline_model/model.py)
* **Our Model** 
   * [Custom Attention Mechanism](./model/util.py)
   * [LG-Enhanced BART Summarizer](./model/model.py)
* **Link Grammar Interface**
    * [Link Grammar API Documentation](./lg_parser/linkgrammar_docs.md)

### Test the model
  
```bash  
  
git clone https://github.com/FrancisDoran/LG-Summarizer.git  
  
cd LG-Summarizer/build/python_deps/  
  
./init.sh && source venv/bin/activate && pip install -r requirements.txt

cd ../..  
  
python3 -m model.model
```
