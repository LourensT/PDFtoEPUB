import os
from mistralai import Mistral, SystemMessage, UserMessage
from mistralai.extra import response_format_from_pydantic_model
from pathlib import Path

import logging
import asyncio
import json
import yaml

from models.metadata import EPUBMetadata
from models.translation import MarkdownTranslation

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ["MISTRAL_API_KEY"]
CONCURRENCY = 25 # number of concurrent requests
NUMBER_OF_CHARACTERS_PER_CHUNK = 4000  # approximate


def translate_metadata(metadata_yaml_fp: Path, target_language: str) -> Path:
    """
    Translate EPUBMetadata to the target language using Mistral API.
    """
    mistral = Mistral(api_key=API_KEY)

    with metadata_yaml_fp.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    
    messages = [
        SystemMessage(content=f"You are a helpful assistant that translates book metadata to {target_language}."),
        UserMessage(content=f"Of the following JSON, translate the translatable values to {target_language}. IGNORE the 'type' fields: \n\n {json.dumps(metadata, indent=2)}"),
    ]

    res = mistral.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        response_format=response_format_from_pydantic_model(EPUBMetadata),
        stream=False,
        temperature=0,
    )

    translated_metadata = res.choices[0].message.content
    output_fp = metadata_yaml_fp.parent / f"metadata_translated_{target_language}.yaml"
    with open(output_fp, "w", encoding="utf-8") as f:
        yaml.dump(json.loads(translated_metadata), f)
    
    return output_fp

def translate_markdown(md_fp: Path, temp_dir: Path, target_language: str) -> str:
    """
    Translate a markdown file to the target language using Mistral API.

    returns the path to the translated markdown file.
    """

    # load markdown content as string
    with md_fp.open("r", encoding="utf-8") as f:
        md_content = f.read()

    logging.info(f"Total characters in markdown: {len(md_content)}")
    logging.info("Splitting into chunks...")
    
    chunks = split_in_chunks(md_content)
    logging.info(f"Total chunks to translate: {len(chunks)}")
    logging.info(f"Largest chunk size: {max([len(c) for c in chunks])}")
    logging.info(f"Smallest chunk size: {min([len(c) for c in chunks])}")
    logging.info(f"Average chunk size: {sum([len(c) for c in chunks]) / len(chunks)}")

    # translate chunks asynchronously
    translated_chunks = asyncio.run(batch_translate_chunks(chunks, target_language))

    # combine translated chunks
    full_translation = join_translated_chunks(translated_chunks)

    output_fp = temp_dir / f"{md_fp.stem}_translated_{target_language}.md"
    with output_fp.open("w", encoding="utf-8") as f:
        f.write(full_translation)

    return output_fp

def split_in_chunks(md_content: str) -> list[str]:
    """
    Split markdown content into chunks. We want to split by paragraphs to preserve formatting.
    """
    chunks = []
    chunk_so_far = ""
    for paragraph in md_content.split("\n\n"):
        if len(chunk_so_far) > NUMBER_OF_CHARACTERS_PER_CHUNK:
            chunks.append(chunk_so_far)
            chunk_so_far = paragraph 
        else:
            if len(paragraph) > NUMBER_OF_CHARACTERS_PER_CHUNK:
                chunks.append(chunk_so_far)
                chunks.append(paragraph)
                chunk_so_far = ""
            else:
                chunk_so_far += "\n\n" + paragraph
            
    chunks.append(chunk_so_far)

    return chunks


def join_translated_chunks(translated_chunks: list[MarkdownTranslation]) -> str:
    """
    Join translated chunks into a single markdown string.
    """
    full_translation = "\n\n".join([chunk["translation"].replace("<STARTCHUNK>", "").replace("<ENDCHUNK>", "").replace("\\n", "\n") for chunk in translated_chunks])
    return full_translation


async def batch_translate_chunks(chunks: list[str], target_language: str) -> list[MarkdownTranslation]:
    """
    Translate a batch of markdown chunks asynchronously.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    counter = {"done": 0, "total": len(chunks)}

    async with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
        tasks = [asyncio.create_task(translate_chunk(mistral, p, sem, counter, target_language)) for p in chunks]
        results = await asyncio.gather(*tasks)

    return results


async def translate_chunk(mistral, prompt, sem, counter, target_language, temp=0):
    async with sem:  # limits concurrent requests
        messages = [
            SystemMessage(content=f"You are a helpful assistant that translates markdown content to {target_language}. Always wrap the output in <STARTCHUNK> and <ENDCHUNK> tags. Preserve all markdown formatting, including whitespace, headings, lists, links, images, bold, italics, code blocks, etc."),
            UserMessage(content=f"Translate the following markdown content to {target_language} while keeping formatting:\n\n <STARTCHUNK>" + prompt + "<ENDCHUNK>"),
        ]

        res = await mistral.chat.complete_async(
            model="mistral-small-latest",
            messages=messages,
            response_format=response_format_from_pydantic_model(MarkdownTranslation),
            stream=False,
            temperature=temp
        )

        counter["done"] += 1
        print(f"Translated {counter['done']} out of {counter['total']} chunks.")

        try:
            # validate response
            return json.loads(res.choices[0].message.content)
        except json.JSONDecodeError as e:
            logging.error("JSON decode error:", exc_info=e)
            logging.error("Response content: %s", res.choices[0].message.content)
            return await translate_chunk(mistral, prompt, sem, counter, target_language, temp=min(temp+0.1, 1))  # retry with increased temperature


if __name__ == "__main__":
    # example usage
    # full_translation = translate_markdown(Path("tmp/forbiddendoor/forbiddendoor.md"), Path("tmp/forbiddendoor"), "French")
    # logging.info(full_translation)
    transl = translate_metadata(Path("tmp/forbiddendoor/metadata.yaml"), "Spanish")
    print(transl)