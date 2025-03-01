{
  description = "Scope Creep";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustPlatform = pkgs.makeRustPlatform {
          rustc = pkgs.rust-bin.stable.latest.default;
          cargo = pkgs.rust-bin.stable.latest.default;
        };
        wasm-bindgen-cli = rustPlatform.buildRustPackage {
          pname = "wasm-bindgen-cli";
          version = "0.2.100";
          useFetchCargoVendor = true;
          src = pkgs.fetchCrate {
            pname = "wasm-bindgen-cli";
            version = "0.2.100";
            sha256 = "sha256-3RJzK7mkYFrs7C/WkhW9Rr4LdP5ofb2FdYGz1P7Uxog=";
          };
          cargoHash = "sha256-qsO12332HSjWCVKtf1cUePWWb9IdYUmT+8OPj/XP2WE=";
        };
      in with pkgs; {
        devShell = mkShell rec {
          buildInputs = [
            # General C Compiler/Linker/Tools
            lld
            clang
            pkg-config
            openssl
            cmake
            (rust-bin.stable.latest.default.override {
              extensions = [ "rust-src" "rust-analysis" ];
              targets = [ "wasm32-unknown-unknown" ];
            })
            rust-analyzer
            cargo-watch
            wasm-bindgen-cli
            zip

            # Graphics and Audio Dependencies
            alsa-lib
            libao
            openal
            libpulseaudio
            udev
            fontconfig
            libxkbcommon
            xorg.libX11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            vulkan-loader
            vulkan-tools
            libGL
            bzip2
            zlib
            libpng
            expat
            brotli
            SDL2
            SDL2_ttf

            # JS/Wasm Deps
            nodejs
            wasm-pack
            binaryen
          ];

          # Allows rust-analyzer to find the rust source
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          # Without this graphical frontends can't find the GPU adapters
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";

          # Needed to build bevy in debug profile
          RUST_MIN_STACK = 33554432;
        };
      });
}
