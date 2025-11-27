# Scanned PDF to EPUB with Mistral OCR
Uses Mistral OCR to convert a scanned PDF to markdown and then to an epub. The main motivation is to be able to read scanned pdfs on my e-reader.

## Install 
1. Install `pandoc` from https://github.com/jgm/pandoc/releases/
2. Make python (3.14) virtual env and install requirements 
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. Create `.env` file with `MISTRAL_API_KEY` variable set to your Mistral API key.

# Usage
```bash
python main.py --pdf_fp path/to/input.pdf --output_fp path/to/output.epub
```

# Costs
Mistral OCR costs depend on the document. My experiments is about 20 cents per 100 pages.

# Approach
1. Use Mistral OCR to get markdown text from each page, with images.
2. Mistral OCR Annotation on the first 10 pages of document for getting book metadata (following )
3. `pandoc` to convert markdown to epub