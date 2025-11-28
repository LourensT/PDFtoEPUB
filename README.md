# Scanned PDF to EPUB with Mistral OCR
Uses Mistral OCR to convert a scanned PDF to markdown and then to an epub. The main motivation is to be able to read scanned pdfs on my e-reader.
There is also an optional automatic translation feature, but the accuracy may vary.

# Install 
1. Install `pandoc` from https://github.com/jgm/pandoc/releases/
    * check installation with `pandoc --version`
2. Make python 3.14 virtual env and install requirements 
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. Set `MISTRAL_API_KEY` environment variable to your Mistral API key.

# Usage
Converting a PDF to EPUB.
```bash
python main.py path/to/input.pdf path/to/output.epub
```
Optional: translate to target language (e.g., French):
```bash
python main.py path/to/input.pdf path/to/output.epub --translate_to french
```
The language parameter is flexible, e.g., `fr`, `fr-FR`, `french`, etc. should all work. 

# API Costs
Costs depend on the document. In my experiments it is about 20 cents per 100 pages for OCR, and about 3-5 cents per 100 pages for translation.

# Approach
1. Mistral OCR to get markdown text from each page, including images.
2. Mistral OCR Annotation on the first 10 pages of document for getting book metadata following [pandoc's epub metadata](https://pandoc.org/demo/example33/11.1-epub-metadata.html).
3. Optional: use mistral-small-latest to translate the markdown in chunks (concurrently for speed).
4. `pandoc` to convert markdown to epub