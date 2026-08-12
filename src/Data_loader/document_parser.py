from __future__ import annotations

import argparse
import os
import subprocess
import sys
from shutil import which
from pathlib import Path
from traceback import print_exception

import docling
import torch
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
DEFAULT_RESUME_DIR = Path("Data") / "Raw" / "CrawlResume"
_DEFAULT_PARSER: DocumentParser | None = None


def _select_accelerator_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        return AcceleratorDevice.CUDA

    return AcceleratorDevice.AUTO


def _create_pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            num_threads=max(os.cpu_count() or 4, 1),
            device=_select_accelerator_device(),
        )
    )


def _build_converter() -> DocumentConverter:
    pdf_pipeline_options = _create_pipeline_options()
    image_pipeline_options = _create_pipeline_options()

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options,
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_options=image_pipeline_options,
            ),
        },
    )


def _get_triton_version() -> str:
    try:
        import triton

        return triton.__version__
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable ({type(exc).__name__}: {exc})"


def _format_root_cause(exc: BaseException) -> BaseException:
    root_cause = exc

    while getattr(root_cause, "__cause__", None) is not None:
        root_cause = root_cause.__cause__  # type: ignore[assignment]

    return root_cause


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_vs_dev_cmd_path() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
    ]

    vswhere = next((candidate for candidate in candidates if candidate.is_file()), None)
    if vswhere is None:
        return None

    try:
        installation_path = subprocess.check_output(
            [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
            text=True,
        ).strip()
    except Exception:
        return None

    if not installation_path:
        return None

    vs_dev_cmd = Path(installation_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    if vs_dev_cmd.is_file():
        return vs_dev_cmd

    return None


def _import_batch_environment(batch_file: Path, arguments: list[str]) -> None:
    argument_text = " ".join(arguments)
    command_text = f'call "{batch_file}" {argument_text} >nul && set'
    batch_output = subprocess.check_output(command_text, shell=True, text=True)

    for line in batch_output.splitlines():
        if "=" not in line:
            continue

        name, value = line.split("=", 1)
        os.environ[name] = value


def _find_cuda_root() -> Path | None:
    candidate_texts = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        str(Path(os.environ.get("ProgramFiles", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"),
        str(Path(os.environ.get("ProgramFiles(x86)", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"),
    ]

    for candidate_text in candidate_texts:
        if not candidate_text:
            continue

        candidate_path = Path(candidate_text)
        if candidate_path.is_file():
            candidate_path = candidate_path.parent

        if candidate_path.name.lower().startswith("v") and candidate_path.is_dir():
            return candidate_path

        if candidate_path.is_dir():
            versioned_candidates = sorted(
                (
                    child
                    for child in candidate_path.iterdir()
                    if child.is_dir() and child.name.lower().startswith("v")
                ),
                reverse=True,
            )
            if versioned_candidates:
                return versioned_candidates[0]

    return None


def _bootstrap_windows_build_environment() -> None:
    if os.name != "nt":
        return

    current_cc = os.environ.get("CC")
    current_cxx = os.environ.get("CXX")
    current_cuda_path = os.environ.get("CUDA_PATH")

    if not current_cc or not current_cxx:
        vs_dev_cmd = _get_vs_dev_cmd_path()
        if vs_dev_cmd is not None:
            _import_batch_environment(vs_dev_cmd, ["-arch=amd64", "-host_arch=amd64"])

    cuda_root = _find_cuda_root()
    if cuda_root is not None:
        os.environ["CUDA_PATH"] = str(cuda_root)
        cuda_bin = cuda_root / "bin"
        if cuda_bin.is_dir():
            path_value = os.environ.get("Path", "")
            cuda_bin_text = str(cuda_bin)
            if cuda_bin_text.lower() not in path_value.lower():
                os.environ["Path"] = f"{cuda_bin_text};{path_value}" if path_value else cuda_bin_text

    cl_path = which("cl.exe")
    if cl_path is not None:
        os.environ["CC"] = cl_path
        os.environ["CXX"] = cl_path
    elif current_cc is None or current_cxx is None:
        # Keep the original environment if we could not improve it.
        if current_cuda_path and "CUDA_PATH" not in os.environ:
            os.environ["CUDA_PATH"] = current_cuda_path


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _resolve_path(path_text: str, project_root: Path) -> Path:
    candidate = Path(path_text)

    if candidate.is_absolute():
        return candidate

    resolved_candidate = project_root / candidate
    if resolved_candidate.exists():
        return resolved_candidate

    return candidate


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _collect_resume_files(resume_dir: Path) -> list[Path]:
    return sorted(
        (
            file_path
            for file_path in resume_dir.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda file_path: file_path.relative_to(resume_dir).as_posix().lower(),
    )


def _print_runtime_diagnostics(parser: DocumentParser) -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: unavailable")

    print(f"Docling: {docling.__version__}")
    print(f"Triton: {_get_triton_version()}")

    converter = parser.converter
    pdf_options = converter.format_to_options[InputFormat.PDF].pipeline_options
    print(f"Docling accelerator: {pdf_options.accelerator_options.device}")


class DocumentParser:
    """Parse resume documents using Docling."""

    def __init__(self) -> None:
        self.converter = _build_converter()

    def parse(self, file_path: str) -> str:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        try:
            result = self.converter.convert(str(path))
            document = result.document

            if document is None:
                raise RuntimeError(f"Docling did not return a document for '{path}'")

            markdown = document.export_to_markdown()
            return markdown.strip()
        except Exception as exc:
            raise RuntimeError(f"Cannot process document '{path}': {exc}") from exc


def get_document_parser() -> DocumentParser:
    global _DEFAULT_PARSER

    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = DocumentParser()

    return _DEFAULT_PARSER


def parse_document(file_path: str) -> str:
    return get_document_parser().parse(file_path)


def _process_file(parser: DocumentParser, file_path: Path, project_root: Path) -> bool:
    print("\n" + "=" * 80)
    print(f"Processing: {_display_path(file_path, project_root)}")
    print("=" * 80)

    try:
        text = parser.parse(str(file_path))
        print(text)
        return True
    except Exception as exc:
        print(f"CV: {file_path.name}")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Message: {exc}")

        root_cause = _format_root_cause(exc)
        if root_cause is not exc:
            print(f"Root cause: {type(root_cause).__name__}: {root_cause}")

        print("Traceback:")
        print_exception(type(exc), exc, exc.__traceback__)
        return False


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()

    parser = argparse.ArgumentParser(
        description="Parse CV documents in Data/Raw/CrawlResume using Docling."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Parse a single document instead of the full CrawlResume directory.",
    )
    args = parser.parse_args(argv)

    project_root = _project_root()
    resume_dir = project_root / DEFAULT_RESUME_DIR

    _bootstrap_windows_build_environment()
    document_parser = get_document_parser()

    print(f"Project root: {project_root}")
    print(f"Resume directory: {resume_dir}")
    _print_runtime_diagnostics(document_parser)

    if not resume_dir.exists():
        print("CrawlResume folder does not exist.")
        return 1

    if args.file:
        file_path = _resolve_path(args.file, project_root)
        resume_files = [file_path]
    else:
        resume_files = _collect_resume_files(resume_dir)

    if not resume_files:
        print("No CV files found.")
        return 1

    print(f"Found {len(resume_files)} CV file(s).")

    success_count = 0
    failure_count = 0

    for resume_file in resume_files:
        if _process_file(document_parser, resume_file, project_root):
            success_count += 1
        else:
            failure_count += 1

    print("\n" + "=" * 80)
    print(f"Tổng số CV: {len(resume_files)}")
    print(f"Thành công: {success_count}")
    print(f"Thất bại: {failure_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())