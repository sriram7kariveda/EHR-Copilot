"""EHR Copilot - Patient Scoped EHR Copilot with Verifiable Evidence."""

import os

# Prevent OpenMP SIGSEGV crash from faiss-cpu + PyTorch BLAS conflict on macOS ARM.
# Must be set before either library is imported.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "0.1.0"
