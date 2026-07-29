"""
Interactive command-line interface for the RAG application.
"""

from __future__ import annotations

from pathlib import Path

from rag.application import RAGApplication
from rag.logger import get_logger
from rag.utils import print_sources

logger = get_logger(__name__)


class CLI:
    """
    Interactive CLI for the Production RAG application.
    """

    def __init__(self) -> None:
        self.app = RAGApplication()
        self.metadata_filter: dict[str, str] | None = None

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def initialize(self) -> None:
        if self.app.vector_manager.database_exists():
            print("\n📂 Loading existing Chroma database...\n")
            logger.info("Loading existing Chroma database.")
            self.app.load_existing()
            return

        print("\nNo existing database found.\n")
        print("1. Build Production Knowledge Base")
        print("2. Exit\n")

        choice = input("Choice: ").strip()

        if choice != "1":
            raise SystemExit

        docs_directory = Path("data/docs")

        if not docs_directory.exists():
            print("\nMissing directory: data/docs\n")
            raise SystemExit

        markdown_files = sorted(docs_directory.glob("*.md"))

        if not markdown_files:
            print("\nNo Markdown files found.\n")
            raise SystemExit

        sources: list[dict] = []

        for markdown in markdown_files:
            sources.append({"type": "markdown", "path": str(markdown)})

        sources.append({"type": "web", "url": "https://thetechfury.com/"})

        # Fixed: line split to under 88 characters
        print(
            f"\nBuilding knowledge base from {len(markdown_files)} "
            f"Markdown document(s) and TechFury website...\n"
        )

        self.app.initialize_sources(sources)

    # ---------------------------------------------------------
    # Menu
    # ---------------------------------------------------------

    def menu(self) -> None:
        print("\nProduction RAG Ready.")
        print("Type 'help' for commands.\n")

        while True:
            try:
                command = input("RAG> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!\n")
                break

            if not command:
                continue

            command_lower = command.lower()

            if command_lower in {"quit", "exit"}:
                print("\nGoodbye!\n")
                break

            if command_lower == "help":
                self.help()
                continue

            if command_lower == "stats":
                print(f"\nIndexed chunks: {self.app.database_size()}\n")
                continue

            if command_lower == "reset":
                confirm = input("Delete Chroma database? (y/n): ").strip().lower()
                if confirm == "y":
                    self.app.reset_database()
                    print("\nDatabase removed.\n")
                    break
                continue

            # ---------------------------------------------
            # use <source>
            # ---------------------------------------------
            if command_lower.startswith("use "):
                source = command[4:].strip()
                self.metadata_filter = {"source": source}
                print(f"\nUsing only:\n{source}\n")
                continue

            # ---------------------------------------------
            # use all
            # ---------------------------------------------
            if command_lower == "use all":
                self.metadata_filter = None
                print("\nSearching all indexed documents.\n")
                continue

            try:
                result = self.app.ask_with_sources(
                    question=command,
                    metadata_filter=self.metadata_filter,
                )
                print("\nAnswer\n")
                print(result["answer"])
                print_sources(result["documents"])
            except Exception as exc:
                logger.exception(exc)
                print(f"\nError: {exc}\n")

    # ---------------------------------------------------------
    # Help
    # ---------------------------------------------------------

    @staticmethod
    def help() -> None:
        print()
        print("Commands")
        print("-------------------------------------------")
        print("help                - Show commands")
        print("stats               - Show indexed chunks")
        print("reset               - Delete Chroma database")
        print("use <source>        - Search one source")
        print("use all             - Search all sources")
        print("exit                - Quit application")
        print()

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self) -> None:
        try:
            self.initialize()
            self.menu()
        except KeyboardInterrupt:
            print("\n\nApplication interrupted.\n")
        except Exception as exc:
            logger.exception(exc)
            print(f"\nFatal error: {exc}\n")
