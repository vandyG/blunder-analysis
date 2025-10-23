{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    buildInputs = [
    pkgs.python314
    pkgs.python311
    pkgs.python312
    pkgs.python311
    pkgs.python310
    pkgs.uv
  ];
}