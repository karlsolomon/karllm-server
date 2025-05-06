import os
import shutil
from typing import List

from fastapi import APIRouter, File, UploadFile
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pydantic import BaseModel

uploaded_file_context = ""

router = APIRouter(prefix="/file")


class UploadPayload(BaseModel):
    content: str


async def upload_file(file: UploadFile, contents: List[str]):
    print("reading")
    file_context = await file.read()
    print("decoding")
    text = file_context.decode("utf-8")
    print("adding to contents list")
    contents.append(f"# File: {file.filename}\n{text}\n#EOF:{file.filename}\n")


async def upload_string(name: str, text: str, contents: List[str]):
    print("uploading pdf text")
    contents.append(f"# File: {name}\n{text}\n#EOF:{name}\n")


TEMP_DIR = "./"


@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    global uploaded_file_context
    contents = []
    input_dir = os.path.join(TEMP_DIR, "pdf_in")
    output_dir = os.path.join(TEMP_DIR, "md_out")
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    converter = PdfConverter(artifact_dict=create_model_dict())

    try:
        for file in files:
            print(f"Saving {file.filename}.")
            if file.filename.endswith(".pdf"):
                file_path = os.path.join(input_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
            else:
                await upload_file(file, contents)
        for file in os.listdir(input_dir):
            print("converting pdf to md")
            config = {
                "output_format": "markdown",
                "disable_image_extraction": True,
            }
            config_parser = ConfigParser(config)
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )
            text, _, _ = text_from_rendered(converter(os.path.join(input_dir, file)))
            print("uploading md")
            await upload_string(f"{file.split(".")[0]}.md", text, contents)
    except Exception as e:
        print(f"Exception in file upload!!! {e}")
    uploaded_file_context = "\n\n".join(contents)
    return {
        "message": f"{len(files)} file(s) uploaded successfully",
        "filenames": [f.filename for f in files],
    }
