from pydantic import BaseModel, Field

class MarkdownTranslation(BaseModel):
    translation: str = Field(
        ...,
        description="Translated version of the source text, preserving meaning, markdown formatting and whitespace. Wrap the entire translation in <STARTCHUNK> and <ENDCHUNK> tags."
    )
    language_from: str = Field(
        ...,
        description="Source language code."
    )
    language_to: str = Field(
        ...,
        description="Target language code."
    )
