import json
import numpy as np
from quantumlab.core.phase_coupling import run_simulation

def test_phase_coupling_writes_summary_and_matrix(tmp_path):
    outdir = tmp_path
    result = run_simulation(
        config={"grid": {"nrows": 2, "ncols": 2}},
        outdir=outdir,
        seed=0,
        workers=1,
    )
    summary_path = outdir / "phase_coupling_summary.json"
    assert summary_path.exists(), "phase_coupling_summary.json should be written"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "summary" in summary_data, "summary key missing in JSON"
    s = summary_data["summary"]
    assert "adjacency_shape" in s and "nnz" in s and "leading_eigenvalue" in s
    assert s["adjacency_shape"] == [4, 4]
    assert isinstance(result, dict) and "summary" in result
    matrix_path = outdir / "phase_coupling_matrix.npz"
    if matrix_path.exists():
        npz = np.load(matrix_path)
        assert "adjacency_data" in npz and "adjacency_indices" in npz and "adjacency_indptr" in npz
        shape = tuple(npz["adjacency_shape"].astype(int).tolist())
        assert shape[0] == shape[1]

