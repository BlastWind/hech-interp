{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Infer where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Maybe
import Data.ByteString.Char8 (pack)
import Data.Constraint
import Data.Proxy
import GHC.Int (Int64)
import GHC.TypeLits
import GPT2.CachedModel
import GPT2.Loader
import SafeTensors hiding (shape)
import System.Environment (getArgs)
import Tiktoken (r50k_base, toRanks)
import qualified Torch as UT
import qualified Torch.DType as D
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen.Managed
import Torch.Typed hiding (length, sample, transformerLM)
import Unsafe.Coerce (unsafeCoerce)
import Data.Dependent.Map ((!))

type family MultinomialCheck (n :: Nat) (shape :: [Nat]) (dim :: Nat) (sat :: Maybe Nat) (result :: Maybe a) :: a where
  MultinomialCheck _ shape dim _ Nothing = DimOutOfBound shape dim
  MultinomialCheck _ shape dim Nothing _ = DimOutOfBound shape dim
  MultinomialCheck n shape dim (Just v) (Just result) = If (n <=? v) result (TypeError (Text "n must be less than or equal to the number of elements in the first dim."))

type Multinomial n shape dim = MultinomialCheck n shape dim (ExtractDim dim shape) (ReplaceDim dim shape n)

multinomial ::
  forall samples dim shape shape' device.
  ( KnownNat samples,
    KnownNat dim,
    shape' ~ Multinomial samples shape dim,
    KnownDevice device
  ) =>
  Tensor device 'Float shape ->
  IO (Tensor device 'Int64 shape')
multinomial t' = cast2 ATen.Managed.multinomial_tl t' (natValI @samples)

sample ::
  forall numTokens batchSize shape device dtype.
  ( All KnownNat [batchSize, numTokens],
    shape ~ [batchSize, numTokens, VocabSize],
    dtype ~ D.Float,
    StandardFloatingPointDTypeValidation device dtype,
    KnownDevice device
  ) =>
  Tensor device dtype shape ->
  IO (Tensor device D.Int64 '[batchSize, 1])
sample tensor' = do
  ix <- multinomial @1 @1 topk_probs
  return $ gatherDim @1 ix topk_indices
  where
    logits = selectIdx @1 tensor' $ fromIntegral $ natValI @numTokens - 1
    (topk_probs, topk_indices) = topk @10 @1 True True $ softmax @1 logits

mkNumTokensProof ::
  forall (numTokens :: Nat).
  (KnownNat numTokens) =>
  Data.Proxy.Proxy numTokens ->
  Maybe (Dict ((1 <=? numTokens) ~ 'True))
mkNumTokensProof Proxy =
  let numEmbeds = natValI @numTokens
   in if numEmbeds > 0
        then Just (unsafeCoerce (Dict :: Dict ('True ~ 'True)))
        else Nothing

infer ::
  forall (numTokens :: Nat).
  (KnownNat numTokens) =>
  Dict ((1 <=? numTokens) ~ 'True) ->
  Model ->
  [[Int64]] ->
  (Tensor ModelDevice UT.Float '[1, numTokens, VocabSize], GPT2ActivationCache ModelDevice 1 numTokens)
infer Dict model tokens =
  transformerLM model
    $ UnsafeMkTensor
      @ModelDevice
      @D.Int64
      @'[1, numTokens]
    $ UT.asTensor tokens

-- | returns the next token prediction, already sampled
-- [1, 1] indicates a batch size of 1 and a sequence length of 1
runInference :: FilePath -> MaybeT IO (Tensor ModelDevice 'Int64 [1, 1])
runInference [] = hoistMaybe Nothing
runInference (fp :: FilePath) = do
  st <- lift $ readSafeTensors fp
  model <- loadGPT2FromSafeTensors st
  tokens <- hoistMaybe $ toRanks r50k_base (pack "ab ab ab ab ab ab")
  withNat (length tokens) $ \(proxy :: Proxy numTokens) ->
    do
      dict <- hoistMaybe $ mkNumTokensProof @numTokens proxy
      let (result, cache) = infer @numTokens dict model [Prelude.map fromIntegral tokens]
          a = cache ! Blocks
      lift $ sample result

-- | Example: `cabal run -- /Users/jane.doe/model.safetensors`, must be absolute!
main :: IO ()
main = do
  args <- getArgs
  _ <- runMaybeT $ runInference (head args)
  return ()
