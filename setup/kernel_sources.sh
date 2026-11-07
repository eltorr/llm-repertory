#!/bin/bash
# A fingerprint of the sources a build compiles, so the setup check can tell compiled kernels older than their
# sources (30_build.sh records it in .repertory/). Plain file reads, nothing imported.
#   kernel_sources.sh ENGINE_DIR offload     the offload extensions engines/exl3_offload/build_ext.py builds: the CUDA
#                                            sources inside expert_cache.py and nv_tier.py (_SRC), and kernels/nv2
#   kernel_sources.sh ENGINE_DIR exllamav3   exllamav3's extension sources
R=${1:?engine checkout}
case "${2:-offload}" in
  offload)
    O="$R/glm53-flash-offload"
    {
      for f in expert_cache nv_tier; do
        awk '/^_SRC = r"""/ { on = 1; next } on && /^"""/ { exit } on' "$O/glm53/$f.py"
      done
      find "$O/kernels/nv2" -type f | LC_ALL=C sort | xargs cat
    } | sha256sum | cut -c1-16 ;;
  exllamav3)
    find "$R/exllamav3/exllamav3/exllamav3_ext" -type f | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16 ;;
  *) echo "usage: $0 ENGINE_DIR offload|exllamav3" >&2; exit 2 ;;
esac
