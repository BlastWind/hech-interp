{
  nixConfig = {
    bash-prompt = "[hech-interp]$ ";
  };
  inputs = {
    utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?rev=01d7c7caba0f021e986f7e46fae3c8e41265a145";
  };

  outputs = { self, nixpkgs, utils  }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
          config.ihaskell.packages = pkgs: with pkgs; [
            hasktorch
          ];
        };
        
        # Create a custom Haskell package set to ensure consistent versions
        haskellPackages = pkgs.haskell.packages.ghc984;        
        # Use the custom package set for GHC and dependencies
        ghcWithHasktorch = haskellPackages.ghcWithPackages (pkgs: with pkgs; [
          hasktorch
          libtorch-ffi
          safe-exceptions
          aeson
          binary
          bytestring
          containers
        ]);
        
        # Create a separate HLS derivation using the same package set
        hls = haskellPackages.haskell-language-server;
        
      in {
        defaultPackage = haskellPackages.callCabal2nix "hech-interp" ./. {};
        devShell = with pkgs; mkShell {
          buildInputs = [
            ghcWithHasktorch
            hls  # Use the HLS from the same package set
            cabal-install
            stack
            ihaskell
            
            # Add build tools
            gcc
            gnumake
            pkg-config
            git
            pcre
          ];
          shellHook = ''
            source ${git}/share/bash-completion/completions/git-prompt.sh
            # Ensure HLS uses the GHC from this environment
            export HIE_BIOS_GHC="${ghcWithHasktorch}/bin/ghc"
            export HIE_BIOS_GHC_ARGS=""
          '';
        };
      });
}
