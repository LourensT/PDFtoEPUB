import argparse
import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

import pdf2image
import yaml
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pypdf import PdfReader

from models.metadata import EPUBMetadata

api_key = os.environ["MISTRAL_API_KEY"]
mistral = Mistral(api_key=api_key)


def main(pdf_fp: str, output_fp: str) -> None:
    """
    Convert a scanned PDF to epub
    """
    # set new logging file
    assert ".pdf" in pdf_fp, "The input file must be a PDF file."
    assert ".epub" in output_fp, "The output file must be an EPUB file"

    pdf_fp = Path(pdf_fp)
    output_fp = Path(output_fp)

    check_pandoc_installed()

    temp_dir = Path(__file__).parent / "tmp" / pdf_fp.stem
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    # set logging
    logging_file = f"{pdf_fp.stem}.log"
    i = 0
    while (temp_dir / logging_file).exists():
        logging_file = f"{pdf_fp.stem}_{i}.log"
        i += 1
    logging_path = temp_dir / logging_file
    logging.basicConfig(filename=logging_path, level=logging.INFO)


    # make output_fp
    output_fp = Path(output_fp)
    if not output_fp.parent.exists():
        output_fp.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Made output dir", output_fp.parent)

    # how long is the book?
    length_of_book = get_num_pages(pdf_fp)
    logging.info(f"Number of pages: {length_of_book}")

    # get metadata from book
    metadata_yaml_fp = pdf_to_metadata(pdf_fp, temp_dir) # stores temp_dir/metadata.yaml

    # do ocr
    # md_path = pdf_to_markdown(pdf_fp, temp_dir) # stores md and images in temp_dir
    md_path = temp_dir / "forbiddendoor.md"

    # create epub using pandoc
    markdown_to_epub(md_path, output_fp, metadata_yaml_fp)


def check_pandoc_installed():
    """
    Checks if pandoc is installed
    """
    output = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
    assert output.stdout[:6] == "pandoc", f"pandoc appears not to be installed, `pandoc --version` gave {output.stdout}"


def get_num_pages(pdf_file: Path) -> int:
    reader = PdfReader(pdf_file)
    return len(reader.pages)


def get_first_page_as_image(pdf_file: Path) -> bytes:
    """
    Use pdf2image to get the first page of the pdf as an image in bytes
    """
    images = pdf2image.convert_from_path(pdf_file, first_page=1, last_page=1)
    return images[0]


def encode_pdf_b64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


def pdf_to_markdown(pdf_fp: Path, output_dir: Path):
    """
    Convert a scanned PDF file to markdown text and save it to the output directory.
     
    Args:
        pdf_fp (Path): The path to the scanned PDF file.
        output_dir (Path): The directory to save the output markdown file and images.
    """

    # raise NotImplementedError("This function is not implemented yet.")


    base64_pdf = encode_pdf_b64(pdf_fp)

    ocr_response = mistral.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        },
        include_image_base64=True,
    )

    all_mkdwn = ""
    for page in ocr_response.pages:

        if len(page.markdown) == 0:
            continue

        if page.markdown.startswith("#"):
            all_mkdwn = all_mkdwn + "\n\n" + page.markdown
        else:
            all_mkdwn = all_mkdwn + "\n" + page.markdown

        # store any images
        for image in page.images:
            image_string = image.image_base64.split(",")[-1]
            image_data = base64.b64decode(image_string)

            image_path = output_dir / image.id
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)

    with (output_dir / pdf_fp.stem).open("w") as f:
        f.write(all_mkdwn)


def pdf_to_metadata(pdf_fp: Path, output_dir: Path) -> Path:
    """
    Extract metadata from a scanned PDF file using OCR.
    Stores metadata as .yaml in TEMP_DIR
    Also saves cover as image in TEMP_DIR

    Args:
        pdf_fp (Path): The path to the scanned PDF file.
    
    Return: 
        path to metadata yaml à la https://pandoc.org/demo/example33/11.1-epub-metadata.html 
    """
    # store cover
    logging.info("Saving first page as cover")
    cover = get_first_page_as_image(pdf_fp)
    # save in temp
    cover.save(output_dir / "cover.png")

    metadata = metadata_annotation(pdf_fp)

    metadata["cover-image"] = str(output_dir / "cover.png")

    # save in temp
    logging.info("Saving metadata.yaml")
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    return output_dir / "metadata.yaml"


def metadata_annotation(pdf_fp: Path) -> Dict:
    base64_pdf = encode_pdf_b64(pdf_fp)

    annotation_response = mistral.ocr.process(
        model="mistral-ocr-latest",
        pages=list(range(8)), # only take first 8 pages
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        },
        document_annotation_format=response_format_from_pydantic_model(EPUBMetadata),
        include_image_base64=False
    )

    return json.loads(annotation_response.document_annotation)


def markdown_to_epub(markdown_fp: Path, output_fp: Path, metadata_yaml_fp: Path = None ) -> None:
    """
    Convert a markdown file to an EPUB file using pandoc.

    Args:
        markdown_fp (Path): The path to the markdown file.
        output_fp (Path): The path to save the output EPUB file.
        metadata_yaml_fp (Path): The path to the metadata YAML file.
    """
    logging.info(f"Using pandoc to convert {markdown_fp} into {output_fp}")
    assert markdown_fp.exists(), "did not find markdown file"

    if not metadata_yaml_fp:
        output = subprocess.run(["pandoc", str(markdown_fp), "-o", str(output_fp)], capture_output=True, text=True)
    else:
        output = subprocess.run(["pandoc", str(markdown_fp), "-o", str(output_fp), "--metadata-file", str(metadata_yaml_fp)], capture_output=True, text=True)

    if len(output.stdout) != 0:
        logging.error(f"Pandoc: {output.stdout}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a scanned PDF file to text.')
    # input file
    parser.add_argument('pdf_fp', type=str, help='The scanned PDF file to convert to text.')
    # output file
    parser.add_argument('output_fp', type=str, help='The output file to save the text to.')

    args = parser.parse_args()
    main(args.pdf_fp, args.output_fp)
