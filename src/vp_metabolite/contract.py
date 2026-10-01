"""A ranked candidate structure carries its own score column."""

SIGNATURE_VERSION = 1
INPUTS = ["smiles"]
OUTPUTS = [
    {
        "name": "metabolite_score",
        "dtype": "float32",
        "range": [0.0, 1.0],
        "semantics": "Forest ranking score for a proposed metabolite; not a formation probability",
        "missing": "An invalid input or a molecule with no generated candidates has no rows",
    }
]


def as_manifest_table() -> dict:
    return {"version": SIGNATURE_VERSION, "inputs": INPUTS, "outputs": OUTPUTS}
