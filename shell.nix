{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}
  '';
  
    buildInputs = [
    pkgs.python314
    pkgs.python313
    pkgs.python312
    pkgs.python311
    pkgs.python310
    pkgs.postgresql
    pkgs.stockfish
  ];
}