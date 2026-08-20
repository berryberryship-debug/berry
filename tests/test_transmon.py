from quantumlab.cli import run_transmon, run_soliton, run_phase_coupling

def test_transmon(tmp_path):
    outdir = tmp_path / "transmon"
    outdir.mkdir()

    result = run_transmon(outdir=outdir)

    assert result.exit_code == 0
    assert (outdir / "spectrum.csv").exists()
    assert (outdir / "metadata.json").exists()

def test_soliton(tmp_path):
    outdir = tmp_path / "soliton"
    outdir.mkdir()

    result = run_soliton(outdir=outdir)

    assert result.exit_code == 0
    assert (outdir / "trajectory.csv").exists()
    assert (outdir / "metadata.json").exists()

def test_phase_coupling(tmp_path):
    outdir = tmp_path / "phase_coupling"
    outdir.mkdir()

    result = run_phase_coupling(outdir=outdir)

    assert result.exit_code == 0
    assert (outdir / "coupling_matrix.csv").exists()
    assert (outdir / "metadata.json").exists()
