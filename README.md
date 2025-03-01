# Hasktorch Skeleton
Templated from [hasktorch-skeleton](https://github.com/hasktorch/hasktorch-skeleton) and [haskell-template](https://github.com/srid/haskell-template?tab=readme-ov-file).

Note, `ghc 9.8.4` might not be available in a HLS release yet (you can check this in your `ghcup tui`). But it is available on HLS master as of Mar 2025. To use this, just
```
ghcup compile hls --ghc 9.8.4 --git-ref master
```
and everything should work.

## Building

1. Enable the Hasktorch binary cache using Cachix:
   ```sh
   cachix use hasktorch
   ```
   This reduces build time significantly on Linux, didn't work on my macbook when I tested it.
2. Launch a Nix shell, which includes GHC with Hasktorch and Haskell Language Server (hls):
   ```sh
   nix develop # You may need to add --extra-experimental-features "nix-command flakes"
   ```
3. Try `ghci`, or `cabal build` now! Note, `stack` doesn't seem to work, but that's fine, you don't
need to `stack ghci` because the `ghci` is in a nix environment with the packages you already need.
I.e., you can `import Torch` in your `ghci` and it should work.

For VSCode users, in order to get HLS linting support, you can choose to either
1. Launch this repo within the `nix` shell that was opened by `nix develop`, or
2. Install [direnv](https://github.com/direnv/direnv-vscode), call `direnv allow` on this repo. When you reopen this repo, `direnv` will automatically load `.envrc`.

## Apple Specific
Apple uses the MPS backend. There is a `toMPS` function that deals with this. Try this in your `ghci` shell:
```haskell
ghci> Torch.Tensor.toMPS $ Torch.ones' [3,3]
```


## Using IHaskell Notebook

Launch Jupyter Notebook in the nix shell:
```sh
ihaskell-notebook
```
