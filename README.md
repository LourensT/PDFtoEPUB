# Scanned PDF to EPUB with GPT
Uses GPT to convert a scanned PDF to an epub. The main motivation is to be able to read books on an e-reader, and to be able to search and highlight text.
This is called "OCR" (Optical Character Recognition), and it is a common problem in computer vision. The main challenge is that the text is not in a structured format, but rather in an image.

Project is currently in a stage where I use it for myself. The API costs are pretty bad, like a $1 per hundred pages depending on image (November 2025). 

## Approach
Do OCR with GPT-5-mini, with a bit of prompt engineering:
1. Page-by-page OCR (image to markdown)
    * pass along an object with context deduced from the previous pages. Specifically:
        * The book title
        * The author
        * The chapter title
        * The last 200 characters of the previous page.
3. Convert the markdown to epub with Pandoc

## Install (linux/wsl, python3.10)
1. Make venv, and install requirements

    `python3 -m venv env`

    `source env/bin/activate`

    `pip install -r requirements.txt`

2. Install pandoc for writing epubs
    1. download from https://github.com/jgm/pandoc/releases/

3. Create `.env` file with `OPENAI_API_KEY=yourkey`

4. **Optional** Install tesseract engine from https://tesseract-ocr.github.io/tessdoc/Installation.html. 
    * This is not strictly necessary, but it provides a back up in case the OpenAI API refuses the request. This happens when (mistakingly or not) the API thinks the request is violating the terms of service.
    * If you want to install it, you can do it with the following commands:

    `sudo apt install tesseract-ocr`

    `sudo apt install libtesseract-dev`

## Usage
After installation (see before), run the `run.sh` script with the path to the pdf as the first argument

`source run.sh path/to/pdf`

## Todo
- [x] Restructure repo
- [ ] Better book metadata (author, title, etc)
    * You can use calibre for this too
- [ ] Image support
    * currently images are substitued by a placeholder description
- [x] ~~Alternative "expensive" approach  which uses OpenAI API for all OCR.~~ This is now the main approach.
    * Seems to be about 0.003 cents for a 1000x600 image. https://openai.com/api/pricing/
    * Sometimes GPT might block because "violates the terms of service", even if this is not the case. The solution here can be to run the local OCR instead when this happens.
