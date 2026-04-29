"""Page 10: Tests — Run the validation suite and display results."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tests.run_tests import run_all, write_results_file  # noqa: E402

st.title("Validation Tests")
st.markdown(
    "This page runs the test suite that validates each major function of the "
    "data product. Each test exercises one feature (data loading, preprocessing, "
    "modeling, prediction, explanation, reporting, or risk classification) and "
    "reports a PASS/FAIL outcome."
)

if st.button("Run Tests", type="primary"):
    with st.spinner("Running validation suite..."):
        results = run_all()
        out_path = ROOT / "tests" / "test_results.txt"
        n_pass, total = write_results_file(results, out_path)

    if n_pass == total:
        st.success(f"All {total} tests passed.")
    else:
        st.error(f"{n_pass}/{total} tests passed; see failures below.")

    for r in results:
        icon = ":white_check_mark:" if r["status"] == "PASS" else ":x:"
        with st.container():
            st.markdown(f"{icon} **{r['name']}**")
            if r["detail"]:
                st.caption(r["detail"])

    st.download_button(
        "Download test_results.txt",
        out_path.read_text(),
        "test_results.txt",
        "text/plain",
    )
else:
    out_path = ROOT / "tests" / "test_results.txt"
    if out_path.exists():
        st.subheader("Last Test Run")
        st.code(out_path.read_text(), language="text")
