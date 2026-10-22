# Portable control core (no ESP-IDF headers): also compiled on the host by
# tests/test_firmware_core.py together with firmware/test/host/ncp_host.c.
idf_component_register(
    SRCS
        "src/ncp_model.c"
        "src/ncp_policy.c"
        "src/ncp_dynamics.c"
        "src/ncp_ekf.c"
        "src/ncp_ctrl.c"
        "src/ncp_sim.c"
        "src/ncp_excite.c"
    INCLUDE_DIRS "include"
)
# -O2 rather than the IDF default -Os for the per-tick paths; no -ffast-math
# (it would break the bit-level parity with numpy/torch the tests rely on).
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wall -Wextra)
