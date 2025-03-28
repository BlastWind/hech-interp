{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module GPT2.CachedModel where

import Control.Monad
import Data.Constraint.Extras.TH (deriveArgDict)
import Data.Dependent.Map
import Data.Dependent.Map (empty)
import Data.Dependent.Sum ((==>))
import Data.Functor.Identity (Identity (..))
import Data.Kind
import Data.Proxy
import qualified Data.Vector.Sized as V
import Debug.Trace
import GHC.Generics
import GHC.TypeLits
import GPT2.HListExtensions
import System.IO.Unsafe (unsafePerformIO)
import qualified Torch as T
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen.Managed
import Torch.NN (HasForward (..))
import qualified Torch.NN as A
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear, log, trace)
import Torch.Typed.NN.Linear
import Torch.Typed.NN.Normalization
import Torch.Typed.NN.Sparse
import Torch.Typed.Parameter
import Torch.Typed.Tensor
import Prelude hiding (cos, exp, sin)

-- TODO not sure if we actually need this
-- import Data.GADT.Compare.TH (deriveGCompare, deriveGEq)
-- import Data.GADT.Show.TH (deriveGShow)

residual f g x = f x >>= (\x' -> g (x `add` x'))

traceTensor ten = trace (show . T.sliceDim 0 0 5 1 . T.select 0 0 . T.squeezeAll $ toDynamic ten) ten

data
  ActivationCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    (dhead :: Nat)
    (nheads :: Nat)
    (nlayers :: Nat)
    a
  where
  Embed :: ActivationCache device dtype batchSize seqLen dmodel dhead nheads nlayers (Tensor device dtype '[batchSize, seqLen, dmodel])
  PosEmbed :: ActivationCache device dtype batchSize seqLen dmodel dhead nheads nlayers (Tensor device dtype '[batchSize, seqLen, dmodel])
  Blocks :: ActivationCache device dtype batchSize seqLen dmodel dhead nheads nlayers [V.Vector nLayers (DMap (BlockCache device dtype batchSize seqLen dmodel dhead nheads) Identity)]
  LnFinal :: ActivationCache device dtype batchSize seqLen dmodel dhead nheads nlayers (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)

data
  BlockCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    (dhead :: Nat)
    (nheads :: Nat)
    a
  where
  ResidPre :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, dmodel])
  Ln1 :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)
  Attn :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (DMap (AttentionCache device dtype batchSize seqLen dmodel dhead nheads) Identity)
  AttnOut :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, dmodel])
  ResidMid :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, dmodel])
  Ln2 :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)
  MLP :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (DMap (MLPCache device dtype batchSize seqLen dmodel) Identity)
  MLPOut :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, dmodel])
  ResidPost :: BlockCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, dmodel])

data
  AttentionCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    (dhead :: Nat)
    (nheads :: Nat)
    a
  where
  Q :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, nheads, dhead])
  K :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, nheads, dhead])
  V :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, nheads, dhead])
  AttnScores :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, nheads, seqLen, seqLen])
  Pattern :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, nheads, seqLen, seqLen])
  Z :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, nheads, dhead])
  Result :: AttentionCache device dtype batchSize seqLen dmodel dhead nheads (Tensor device dtype '[batchSize, seqLen, nheads, dmodel])

data
  LayerNormCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    a
  where
  Scale :: LayerNormCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, 1])
  Normalized :: LayerNormCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, dmodel])

data
  MLPCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    a
  where
  MLPpre :: MLPCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, 4 * dmodel])
  MLPpost :: MLPCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, 4 * dmodel])

-- TODO not sure if we actually need this it's in the example
-- deriveGEq ''CacheTag
-- deriveGCompare ''CacheTag
-- deriveGShow ''CacheTag
-- deriveArgDict ''CacheTag
type GPT2ActivationMap device batchSize seqLen = DMap (ActivationCache device D.Float batchSize seqLen 768 64 12 12) Identity

geluApproximate ::
  forall shape dtype device.
  (GeluDTypeIsValid device dtype) =>
  Tensor device dtype shape ->
  String ->
  Tensor device dtype shape
geluApproximate _self _approximate = unsafePerformIO $ cast2 ATen.Managed.gelu_ts _self _approximate

--------------------------------------------------------------------------------
-- Relation-Aware Multi-Headed Attention Layer
--------------------------------------------------------------------------------

data
  MultiheadAttentionSpec
    (dmodel :: Nat)
    (nHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttentionSpec ::
    -- | spec for dropout
    MultiheadAttentionSpec dmodel nHeads dtype device
  deriving (Show, Eq)

data
  MultiheadAttention
    (dmodel :: Nat)
    (nHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttention ::
    { -- | packed in-projection for q, k, v
      mhaQInProj :: Linear dmodel dmodel dtype device,
      -- | in-projection for key
      mhaKInProj :: Linear dmodel dmodel dtype device,
      -- | in-projection for value
      mhaVInProj :: Linear dmodel dmodel dtype device,
      -- | out-projection
      mhaOutProj :: Linear dmodel dmodel dtype device
    } ->
    MultiheadAttention dmodel nHeads dtype device
  deriving (Show, Generic, Parameterized)

data
  MultiheadAttentionActivationCache
    (dmodel :: Nat)
    (nHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttentionActivationCache ::
    { -- | The dot product of q, k
      attn_scores :: Tensor device dtype '[batchSize, nHeads, seqLen, seqLen],
      -- | Softmaxed and masked "attn_scores"
      pattern :: Tensor device dtype '[batchSize, nHeads, seqLen, seqLen],
      -- | The value activation weighed by "pattern"
      z :: Tensor device dtype '[batchSize, seqLen, nHeads, dhead],
      -- | z multiplied by the out-projection
      result :: Tensor device dtype '[batchSize, seqLen, nHeads, dmodel]
    } ->
    MultiheadAttentionActivationCache dmodel nHeads dtype device

multiheadAttention ::
  forall dmodel nHeads seqLen batchSize dhead dtype device.
  ( 1 <= nHeads,
    dmodel ~ (dhead * nHeads),
    All KnownNat '[dmodel, dmodel, dmodel, nHeads, seqLen, batchSize, dhead],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | multi-head attention model ADT
  MultiheadAttention dmodel nHeads dtype device ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]) ->
  -- | query representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | key representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | value representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | attention and attention averaged over heads, the `attn_out`.
  ( Tensor device dtype '[batchSize, seqLen, dmodel],
    MultiheadAttentionActivationCache dmodel nHeads dtype device
  )
multiheadAttention MultiheadAttention {..} attentionMask query key value = (attn_out, MultiheadAttentionActivationCache attn_scores pattern z result'')
  where
    scaling = Prelude.sqrt . fromIntegral $ natValI @dhead :: Double
    q = reshape' . forward mhaQInProj $ query
    k = reshape' . forward mhaKInProj $ key
    v = reshape' . forward mhaVInProj $ value
    _maskAttention attentionWeights =
      case attentionMask of
        Nothing -> attentionWeights
        Just am -> attentionWeights `add` unsqueeze @1 am
    attn_scores = _maskAttention $ divScalar scaling $ matmul q (transpose @2 @3 k) -- dot product, scaled (by sqrt, not softmax)
    pattern = softmax @3 attn_scores
    z = transpose @1 @2 $ matmul pattern v
    -- z' is z with the nHeads and seqLen swapped so that `matmul z' wo` is broadcasted correctly.
    z' = reshape @'[batchSize, nHeads, seqLen, dhead] z
    -- KEY: While `mhaOutProj` is efficiently represented with dims [dmodel, dmodel],
    -- its meaning is clearer when seen as [nHeads, dhead, dmodel]. These are equiv because dmodel ~ nHeads * dhead
    -- We need to reshape to the meaningful representation in order to calculate result'', which is the meaningful cache.
    wo = reshape @'[nHeads, dhead, dmodel] $ toDependent $ weight mhaOutProj
    -- TODO: According to [this](https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/transformer-full-updated.png),
    -- this calculation is very expensive, so much so that TransformerLens has a use_attn_result flag to toggle this.
    result' = matmul z' wo
    result'' = reshape @'[batchSize, seqLen, nHeads, dmodel] result'
    attn_out = forward mhaOutProj . reshape @'[batchSize, seqLen, dmodel] $ z
    reshape' ::
      forall seqLen'.
      (KnownNat seqLen') =>
      Tensor device dtype '[batchSize, seqLen', dmodel] ->
      Tensor device dtype '[batchSize, nHeads, seqLen', dhead]
    reshape' t' = transpose @1 @2 $ reshape @'[batchSize, seqLen', nHeads, dhead] t'

instance
  ( All KnownNat '[dmodel, dmodel, dmodel, nHeads],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (MultiheadAttentionSpec dmodel nHeads dtype device)
    (MultiheadAttention dmodel nHeads dtype device)
  where
  sample MultiheadAttentionSpec =
    MultiheadAttention
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec

--------------------------------------------------------------------------------
-- Transformer MLP Layer
--------------------------------------------------------------------------------

data
  TransformerMLPSpec
    (dmodel :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLPSpec ::
    forall dmodel ffnDim dtype device.
    { -- | epsilon for layer norm
      epsSpec :: Double
    } ->
    TransformerMLPSpec dmodel ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerMLP
    (dmodel :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLP ::
    forall dmodel ffnDim dtype device.
    { -- | first fully connected layer
      linear0 :: Linear dmodel ffnDim dtype device,
      -- | second fully connected layer
      linear1 :: Linear ffnDim dmodel dtype device,
      -- | layer norm
      ln :: LayerNorm '[dmodel] dtype device
    } ->
    TransformerMLP dmodel ffnDim dtype device
  deriving (Show, Generic, Parameterized)

transformerMLP ::
  forall dmodel ffnDim maxSeqLen batchSize dtype device.
  ( BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    KnownNat dmodel,
    GeluDTypeIsValid device dtype,
    IsSuffixOf '[dmodel] '[maxSeqLen, batchSize, dmodel]
  ) =>
  -- | MLP model ADT for transformer
  TransformerMLP dmodel ffnDim dtype device ->
  Tensor device dtype '[maxSeqLen, batchSize, dmodel] -> -- input
  (Tensor device dtype '[maxSeqLen, batchSize, dmodel]) -- output
transformerMLP TransformerMLP {..} x =
  (`add` x)
    . forward linear1
    . (`geluApproximate` "tanh")
    . forward linear0
    $ forward ln x

instance
  ( All KnownNat '[dmodel, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerMLPSpec dmodel ffnDim dtype device)
    (TransformerMLP dmodel ffnDim dtype device)
  where
  sample TransformerMLPSpec {..} =
    TransformerMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (LayerNormSpec epsSpec)

--------------------------------------------------------------------------------
-- Relation-Aware Transformer Layer
--------------------------------------------------------------------------------

data
  TransformerLayerSpec
    (dmodel :: Nat)
    (nHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayerSpec ::
    forall dmodel nHeads ffnDim dtype device.
    { mhaSpec :: MultiheadAttentionSpec dmodel nHeads dtype device,
      epsSpec' :: Double,
      mlpSpec :: TransformerMLPSpec dmodel ffnDim dtype device
    } ->
    TransformerLayerSpec dmodel nHeads ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerLayer
    (dmodel :: Nat)
    (nHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayer ::
    forall dmodel nHeads ffnDim dtype device.
    { -- | multi-head attention
      transformerLayer_mha :: MultiheadAttention dmodel nHeads dtype device,
      -- | layer norm
      transformerLayer_ln :: LayerNorm '[dmodel] dtype device,
      -- | MLP
      transformerLayer_mlp :: TransformerMLP dmodel ffnDim dtype device
    } ->
    TransformerLayer dmodel nHeads ffnDim dtype device
  deriving (Show, Generic, Parameterized)

data
  TransformerLayerActivationCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
  where
  TransformerLayerActivationCache ::
    forall device dtype batchSize seqLen dmodel.
    { transformerLayerKeyActivation :: Tensor device dtype '[batchSize, seqLen, dmodel],
      transformerLayerQueryActivation :: Tensor device dtype '[batchSize, seqLen, dmodel],
      transformerLayerValueActivation :: Tensor device dtype '[batchSize, seqLen, dmodel]
    } ->
    TransformerLayerActivationCache device dtype batchSize seqLen dmodel

transformerLayer ::
  forall (nHeads :: Nat) (ffnDim :: Nat) (dmodel :: Nat) (dhead :: Nat) (seqLen :: Nat) (batchSize :: Nat) dtype device.
  ( 1 <= nHeads,
    dmodel ~ (dhead * nHeads),
    All KnownNat '[dmodel, dmodel, dmodel, nHeads, seqLen, batchSize, dhead],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    KnownDType dtype,
    dtype ~ SumDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    GeluDTypeIsValid device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | transformer layer model ADT
  TransformerLayer dmodel nHeads ffnDim dtype device ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]) ->
  -- | query representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | key representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | value representation
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | transformer layer output representation
  (Tensor device dtype '[batchSize, seqLen, dmodel], TransformerLayerActivationCache device dtype batchSize seqLen dmodel)
transformerLayer TransformerLayer {..} attentionMask query key value =
  let key' = forward transformerLayer_ln key
      value' = forward transformerLayer_ln value
      f query' = multiheadAttention transformerLayer_mha attentionMask query' key' value'
   in -- _ <- print . T.sliceDim 0 0 5 1 . T.select 0 0 . T.squeezeAll . toDynamic $ fst r
      do
        let (result, cache) = f (forward transformerLayer_ln query)
        let result' = query `add` result
        (transformerMLP transformerLayer_mlp result', cache)

instance
  ( All KnownNat '[dmodel, dmodel, dmodel, nHeads, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerLayerSpec dmodel nHeads ffnDim dtype device)
    (TransformerLayer dmodel nHeads ffnDim dtype device)
  where
  sample TransformerLayerSpec {..} =
    TransformerLayer
      <$> A.sample mhaSpec
      <*> A.sample (LayerNormSpec epsSpec')
      <*> A.sample mlpSpec

--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data
  GPT2Spec
    (numAttnLayers :: Nat)
    (nHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (dmodel :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2Spec ::
    forall numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device.
    { -- | spec for each and every transformer layer
      lmLayerSpec :: TransformerLayerSpec dmodel nHeads ffnDim dtype device,
      epsSpec'' :: Double
    } ->
    GPT2Spec numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device
  deriving (Show, Eq)

data
  GPT2
    (numAttnLayers :: Nat)
    (nHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (dmodel :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2 ::
    forall numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device.
    { -- | token embedding
      tEmbedding :: Embedding ('Just paddingIdx) vocabSize dmodel 'Learned dtype device,
      -- | positional embedding
      tPosEmbedding :: Embedding 'Nothing maxSeqLen dmodel 'Constant dtype device,
      -- | transformer layers
      tLayers :: HList (HReplicateR numAttnLayers (TransformerLayer dmodel nHeads ffnDim dtype device)),
      -- | final layer norm
      tFinalLN :: LayerNorm '[dmodel] dtype device,
      -- | final output projection
      tProj :: Linear dmodel vocabSize dtype device
    } ->
    GPT2 numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device
  deriving (Generic)

deriving instance
  ( Show
      ( HList
          ( HReplicateR
              numAttnLayers
              ( TransformerLayer
                  dmodel
                  nHeads
                  ffnDim
                  dtype
                  device
              )
          )
      )
  ) =>
  Show (GPT2 numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)

instance
  ( layers
      ~ HReplicateR
          numAttnLayers
          ( TransformerLayer
              dmodel
              nHeads
              ffnDim
              dtype
              device
          ),
    Parameterized
      ( HList
          layers
      ),
    HAppendFD
      (Parameters (HList layers))
      '[ Parameter device dtype '[dmodel],
         Parameter device dtype '[dmodel],
         Parameter device dtype '[vocabSize, dmodel],
         Parameter device dtype '[vocabSize]
       ]
      ( Parameters (HList layers)
          ++ '[ Parameter device dtype '[dmodel],
                Parameter device dtype '[dmodel],
                Parameter device dtype '[vocabSize, dmodel],
                Parameter device dtype '[vocabSize]
              ]
      )
  ) =>
  Parameterized (GPT2 numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)

newtype
  FoldLayers
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = FoldLayers
  { -- | optional attention mask
    flAttentionMask :: Maybe (Tensor device dtype '[batchSize, seqLen, seqLen])
  }

instance
  ( 1 <= nHeads,
    dmodel ~ (dhead * nHeads),
    All KnownNat '[dmodel, nHeads, seqLen, batchSize, dhead],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    GeluDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  ApplyAB
    (FoldLayers batchSize seqLen dtype device)
    ( TransformerLayer dmodel nHeads ffnDim dtype device,
      Tensor device dtype '[batchSize, seqLen, dmodel]
    )
    (Tensor device dtype '[batchSize, seqLen, dmodel])
  where
  applyAB FoldLayers {..} (layer, x) = do
    let (res, cache) = transformerLayer layer flAttentionMask x x x in res

transformerLM ::
  forall
    numAttnLayers
    nHeads
    ffnDim
    paddingIdx
    vocabSize
    dmodel
    seqLen
    maxSeqLen
    batchSize
    dtype
    device.
  ( All KnownNat '[paddingIdx, dmodel, seqLen, batchSize],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    paddingIdx + 1 <= vocabSize,
    1 <= seqLen,
    HLast (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])) (Tensor device dtype '[batchSize, seqLen, dmodel]),
    HScanr
      (FoldLayers batchSize seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, dmodel])
      (HReplicateR numAttnLayers (TransformerLayer dmodel nHeads ffnDim dtype device))
      (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  GPT2 numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device ->
  Tensor device 'D.Int64 '[batchSize, seqLen] ->
  (Tensor device dtype '[batchSize, seqLen, vocabSize], GPT2ActivationMap device batchSize seqLen)
transformerLM GPT2 {..} xTokens = do
  let x = embed tEmbedding xTokens
      positions =
        expand @'[batchSize, seqLen, dmodel] True
          -- . (\pos_emb -> trace (show . T.select 0 0 $ toDynamic pos_emb) pos_emb)
          . embed tPosEmbedding
          . Torch.Typed.Tensor.toDType @D.Int64
          . linspace @seqLen (0 :: Int)
          $ natValI @(seqLen - 1)
  let x' = x `add` positions
  let attentionMask =
        unsqueeze @0
          . Torch.Typed.Tensor.toDType @D.Bool
          . triu 1
          $ ones @'[seqLen, seqLen] @D.Int8 @device
      attentionMask' =
        pure . maskedFill attentionMask (-(1 / 0) :: Double) $
          zeros @'[batchSize, seqLen, seqLen] @dtype @device
  -- _ <- print $ shape x
  -- _ <- print (T.select 0 0 . T.squeezeAll $ toDynamic x)

  -- (\final -> trace (show . T.sliceDim 0 0 5 1 . T.select 0 0 . T.squeezeAll $ toDynamic final) final) $
  -- (\fin -> trace (show . T.select 0 0 . T.squeezeAll $ toDynamic fin) forward tProj fin) $

  let intermediateLayerOutputs :: HList (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel]))
      intermediateLayerOutputs = hScanr (FoldLayers attentionMask') x' tLayers
      l = hLast intermediateLayerOutputs
  (forward tProj $ forward tFinalLN l, empty)

instance
  ( All KnownNat '[paddingIdx, dmodel, seqLen, batchSize, seqLen],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    paddingIdx + 1 <= vocabSize,
    1 <= seqLen,
    HLast (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])) (Tensor device dtype '[batchSize, seqLen, dmodel]),
    HScanr
      (FoldLayers batchSize seqLen dtype device)
      (Tensor device dtype '[batchSize, seqLen, dmodel])
      (HReplicateR numAttnLayers (TransformerLayer dmodel nHeads ffnDim dtype device))
      (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  HasForward (GPT2 numAttnLayers nHeads ffnDim paddingIdx seqLen vocabSize dmodel dtype device) (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, vocabSize])
  where
  forwardStoch model input = return $ fst $ transformerLM model input
  forward model input = fst $ transformerLM model input

sinusoidal ::
  forall vocabSize dmodel device.
  ( All KnownNat '[vocabSize, dmodel],
    1 <= vocabSize,
    1 <= Div dmodel 2,
    (Div dmodel 2 * 2) ~ dmodel,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  Tensor device 'D.Float '[vocabSize, dmodel]
sinusoidal =
  let positions =
        unsqueeze @1
          . linspace @vocabSize (0 :: Int)
          $ natValI @(vocabSize - 1)
      scalingFactors =
        exp
          . mulScalar
            ( -( log (10000 :: Double)
                   / (fromInteger . natVal $ Proxy @(Div dmodel 2))
               )
            )
          . linspace @(Div dmodel 2) (0 :: Int)
          $ natValI @(Div dmodel 2 - 1)
      radians = mul positions scalingFactors
      weights = stack @2 (sin radians :. cos radians :. HNil)
   in reshape weights

instance
  ( paddingIdx <= vocabSize,
    1 <= maxSeqLen,
    1 <= vocabSize - paddingIdx,
    1 <= Div dmodel 2,
    (((vocabSize - paddingIdx) - 1) + (1 + paddingIdx)) ~ vocabSize,
    (Div dmodel 2 * 2) ~ dmodel,
    All KnownNat '[ffnDim, paddingIdx, vocabSize, maxSeqLen, dmodel],
    HReplicate numAttnLayers (TransformerLayerSpec dmodel nHeads ffnDim dtype device),
    A.Randomizable
      (HList (HReplicateR numAttnLayers (TransformerLayerSpec dmodel nHeads ffnDim dtype device)))
      (HList (HReplicateR numAttnLayers (TransformerLayer dmodel nHeads ffnDim dtype device))),
    KnownDType dtype,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  A.Randomizable
    (GPT2Spec numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)
    (GPT2 numAttnLayers nHeads ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)
  where
  sample GPT2Spec {..} =
    GPT2
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample (LayerNormSpec epsSpec'')
      <*> A.sample LinearSpec
