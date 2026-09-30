from lectra.domain.document import Document


def test_document_creation() -> None:
    """Verify a document stores its source information."""
    document = Document(
        id="doc-001",
        name="Operating Systems Notes",
        source_type="pdf",
        metadata={"course": "OS", "module": 3},
    )

    assert document.id == "doc-001"
    assert document.name == "Operating Systems Notes"
    assert document.source_type == "pdf"
    assert document.metadata["module"] == 3
