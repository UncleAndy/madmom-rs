{
  description = "madmom-rs — Rust port of the madmom audio/music signal processing library";

  # NixOS 26.05 to match the development host. Bump together with the host.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forAllSystems (pkgs:
        let
          # Python used only for the tools/ scripts: the reference madmom build,
          # the stage dumps and the ONNX converter. Never a runtime dependency.
          referencePython = pkgs.python3.withPackages (ps: [
            ps.numpy
            ps.scipy
            ps.cython
            ps.pytest
            ps.packaging
            ps.setuptools
          ]);

          # Triton runtime libraries are built out-of-store by
          # tools/build_triton_core.sh; they are NOT provided by nixpkgs.
          shellHook = ''
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            export MADMOM_VERSION="''${MADMOM_VERSION:-0.1.0}"
            # Fixtures: the parity tests read these unless MADMOM_FIXTURES is set.
            export MADMOM_FIXTURES="''${MADMOM_FIXTURES:-$PWD/third_party/madmom/tests/data}"
            if [ -f "$PWD/tools/triton_env.sh" ] && [ -d "$PWD/third_party/triton-runtime/lib" ]; then
              source "$PWD/tools/triton_env.sh"
            fi
            echo "madmom-rs dev shell: $(rustc --version 2>/dev/null || echo 'no rustc')"
            echo "reference python: $(python3 -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)' 2>/dev/null || echo 'not available')"
          '';
        in
        {
          default = pkgs.mkShell {
            name = "madmom-rs";
            packages = with pkgs; [
              cargo
              rustc
              clippy
              rustfmt
              rust-analyzer
              referencePython
              cmake
              gnumake
              pkg-config
              gcc
              llvmPackages.libclang
              ffmpeg
              git
              curl
              openssl
              zlib
              python3
            ];
            inherit shellHook;
          };

          # env for converting models to ONNX and validating them
          # (`cargo` is deliberately absent: this shell is for tools/ only).
          models = pkgs.mkShell {
            name = "madmom-rs-models";
            packages = [ referencePython pkgs.python3 ];
            shellHook = shellHook + ''
              echo "note: tools/convert_to_onnx.py needs the onnx package, which is"
              echo "      not in nixpkgs here; run it as:"
              echo "      nix-shell -p \"python3.withPackages(ps: [ps.numpy ps.onnx])\" --run \\"
              echo "        python3 tools/convert_to_onnx.py --help"
            '';
          };
        });
    };
}
