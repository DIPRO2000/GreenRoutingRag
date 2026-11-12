"""Helpers for unambiguous document selection in the UI."""


def build_document_options(documents):
    """Return UI labels mapped to document IDs without collisions."""
    options = {"All Indexed Documents": None}
    for document in documents:
        label = (
            f"{document['filename']} ({document['total_chunks']} chunks, "
            f"ID {document['doc_id']})"
        )
        options[label] = document["doc_id"]
    return options
