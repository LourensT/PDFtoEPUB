from typing import Optional, Union, List, Literal
from pydantic import BaseModel, Field

# https://pandoc.org/demo/example33/11.1-epub-metadata.html

class Identifier(BaseModel):
    text: Optional[str] = None
    scheme: Optional[Literal[
        "ISBN-10", "GTIN-13", "UPC", "ISMN-10", "DOI", "LCCN",
        "GTIN-14", "ISBN-13", "Legal deposit number", "URN", "OCLC",
        "ISMN-13", "ISBN-A", "JP", "OLCC"
    ]] = None


class Title(BaseModel):
    text: Optional[str] = None
    type: Optional[Literal[
        "main", "subtitle", "short", "collection", "edition", "extended"
    ]] = None
    file_as: Optional[str] = Field(None, alias="file-as")


class Creator(BaseModel):
    text: Optional[str] = None
    role: Optional[Literal["author", "translator"]] = None # all valid MARC contributoors https://loc.gov/marc/relators/relaterm.html
    file_as: Optional[str] = Field(None, alias="file-as")


class Subject(BaseModel):
    text: Optional[str] = None
    authority: Optional[Literal[
        "AAT", "BIC", "BISAC", "CLC", "DDC", "CLIL", "EuroVoc",
        "MEDTOP", "LCSH", "NDC", "Thema", "UDC", "WGS"
    ]] = None
    term: Optional[str] = None


class IBooks(BaseModel):
    version: Optional[str] = None
    specified_fonts: Optional[bool] = Field(None, alias="specified-fonts")
    ipad_orientation_lock: Optional[Literal["portrait-only", "landscape-only"]] = Field(
        None, alias="ipad-orientation-lock"
    )
    iphone_orientation_lock: Optional[Literal["portrait-only", "landscape-only"]] = Field(
        None, alias="iphone-orientation-lock"
    )
    binding: Optional[bool] = None
    scroll_axis: Optional[Literal["vertical", "horizontal", "default"]] = Field(
        None, alias="scroll-axis"
    )


class EPUBMetadata(BaseModel):
    identifier: Optional[Union[Identifier, List[Identifier]]] = None
    title: Optional[Union[str, Title, List[Title]]] = None
    creator: Optional[Union[Creator, List[Creator]]] = None
    contributor: Optional[Union[Creator, List[Creator]]] = None
    date: Optional[str] = None
    lang: Optional[str] = None
    subject: Optional[Union[str, Subject, List[Subject]]] = None
    description: Optional[str] = None
    type: Optional[str] = None
    format: Optional[str] = None
    relation: Optional[str] = None
    coverage: Optional[str] = None
    rights: Optional[str] = None
    belongs_to_collection: Optional[str] = Field(None, alias="belongs-to-collection")
    group_position: Optional[int] = Field(None, alias="group-position")
    page_progression_direction: Optional[Literal["ltr", "rtl"]] = Field(
        None, alias="page-progression-direction"
    )
    ibooks: Optional[IBooks] = None
    publisher: Optional[str] = None
    
    class Config:
        populate_by_name = True  # Allows using both snake_case and hyphenated names