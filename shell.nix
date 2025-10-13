{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python313
    uv
    stdenv.cc.cc.lib
    zlib
    python313Packages.numpy
    python313Packages.pandas
    python313Packages.scipy
    python313Packages.matplotlib
    python313Packages.scikit-learn
    python313Packages.opencv4
    libGL
    glib
    libglvnd
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.glib.out}/lib:${pkgs.libglvnd}/lib:$LD_LIBRARY_PATH"
    echo "NixOS Python environment ready!"
    echo "Run: uv run reformat_dataset.py"
  '';
}
