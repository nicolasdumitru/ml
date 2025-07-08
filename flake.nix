{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true; # Required for CUDA packages
          cudaSupport = true; # Enable CUDA for packages that support it
        };
      };
    in
    {
      devShell."${system}" = pkgs.mkShell {
        packages = with pkgs; [
          (python313.withPackages (
            ps: with ps; [
              # Libraries
              numpy
              torch
              torchvision
              scikit-learn
              scipy
              pandas
              matplotlib

              # Tooling
              python-lsp-server
              jupyter
              kaggle
            ]
          ))

          # Tooling
          pyright
          black
        ];
      };
    };
}
