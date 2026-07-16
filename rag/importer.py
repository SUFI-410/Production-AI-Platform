"""
Production Markdown importer.

Downloads technical documentation from websites and stores it
as Markdown for indexing into the RAG knowledge base.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup

from rag.logger import get_logger

logger = get_logger(__name__)


class MarkdownImporter:
    """
    Downloads documentation websites and converts them into
    markdown files suitable for ingestion.

    Example
    -------
    importer = MarkdownImporter()

    importer.import_website(
        "https://docs.python.org/3/tutorial/",
        max_pages=50,
    )
    """

    def __init__(
        self,
        docs_directory: str | Path = "data/docs",
    ) -> None:

        self.docs_directory = Path(docs_directory)

        self.docs_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "Production-RAG/1.0 "
                    "(https://github.com/)"
                )
            }
        )

        self.converter = html2text.HTML2Text()

        self.converter.ignore_images = True
        self.converter.ignore_links = False
        self.converter.body_width = 0

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def sanitize_filename(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9]+",
            "_",
            text,
        )

        text = text.strip("_")

        return text or "document"

    def save_document(
        self,
        filename: str,
        markdown: str,
    ) -> Path:

        if not filename.endswith(".md"):
            filename += ".md"

        path = self.docs_directory / filename

        path.write_text(
            markdown.strip(),
            encoding="utf-8",
        )

        logger.info(
            "Saved %s",
            path,
        )

        return path

    def page_to_markdown(
        self,
        html: str,
    ) -> str:

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "svg",
                "header",
                "footer",
                "nav",
                "aside",
                "noscript",
                "form",
            ]
        ):
            tag.decompose()

        return self.converter.handle(
            str(soup)
        )
    

    # ---------------------------------------------------------
    # Download Single Page
    # ---------------------------------------------------------

    def download_page(
        self,
        url: str,
        filename: str | None = None,
    ) -> Path:

        logger.info(
            "Downloading %s",
            url,
        )

        response = self.session.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        markdown = self.page_to_markdown(
            response.text,
        )

        if filename is None:

            parsed = urlparse(url)

            filename = (
                parsed.path.strip("/")
                .replace("/", "_")
            )

            if not filename:
                filename = "index"

            filename = self.sanitize_filename(
                filename
            )

        return self.save_document(
            filename,
            markdown,
        )

    # ---------------------------------------------------------
    # Crawl Documentation Website
    # ---------------------------------------------------------

    def import_website(
        self,
        start_url: str,
        max_pages: int = 100,
    ) -> list[Path]:

        parsed = urlparse(start_url)

        domain = parsed.netloc

        visited: set[str] = set()

        queue: list[str] = [
            start_url
        ]

        saved_files: list[Path] = []

        while queue and len(visited) < max_pages:

            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            logger.info(
                "Crawling %s",
                url,
            )

            try:

                response = self.session.get(
                    url,
                    timeout=30,
                )

                response.raise_for_status()

            except Exception as exc:

                logger.warning(
                    "Skipping %s (%s)",
                    url,
                    exc,
                )

                continue

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            markdown = self.page_to_markdown(
                response.text,
            )

            parsed_url = urlparse(url)

            filename = (
                parsed_url.path.strip("/")
                .replace("/", "_")
            )

            if not filename:
                filename = "index"

            filename = self.sanitize_filename(
                filename,
            )

            saved_files.append(
                self.save_document(
                    filename,
                    markdown,
                )
            )

            for link in soup.find_all(
                "a",
                href=True,
            ):

                absolute = urljoin(
                    url,
                    link["href"],
                )

                parsed_link = urlparse(
                    absolute,
                )

                absolute = parsed_link._replace(
                    fragment="",
                    query="",
                ).geturl()

                if parsed_link.netloc != domain:
                    continue

                if absolute in visited:
                    continue

                if absolute not in queue:
                    queue.append(
                        absolute
                    )

        logger.info(
            "Imported %s markdown document(s).",
            len(saved_files),
        )

        return saved_files

    # ---------------------------------------------------------
    # Import Multiple Documentation Sites
    # ---------------------------------------------------------

    def import_websites(
        self,
        urls: list[str],
        max_pages: int = 100,
    ) -> list[Path]:

        imported: list[Path] = []

        for url in urls:

            imported.extend(
                self.import_website(
                    start_url=url,
                    max_pages=max_pages,
                )
            )

        logger.info(
            "Imported %s total document(s).",
            len(imported),
        )

        return imported

    # ---------------------------------------------------------
    # List Documents
    # ---------------------------------------------------------

    def list_documents(
        self,
    ) -> list[Path]:

        return sorted(
            self.docs_directory.glob(
                "*.md"
            )
        )
    

    # ---------------------------------------------------------
    # Delete Documents
    # ---------------------------------------------------------

    def delete_document(
        self,
        filename: str,
    ) -> None:

        if not filename.endswith(".md"):
            filename += ".md"

        path = (
            self.docs_directory
            / filename
        )

        if path.exists():

            path.unlink()

            logger.info(
                "Deleted %s",
                path,
            )

    # ---------------------------------------------------------
    # Delete All Documents
    # ---------------------------------------------------------

    def clear_documents(
        self,
    ) -> None:

        for document in self.list_documents():
            document.unlink()

        logger.info(
            "All markdown documents removed."
        )
