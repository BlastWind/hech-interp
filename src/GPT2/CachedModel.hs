{-# LANGUAGE AllowAmbiguousTypes #-}
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
import Data.GADT.Compare
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
import qualified Torch.Internal.Cast as ATen
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
    (nhead :: Nat)
    (nlayers :: Nat)
    a
  where
  Embed :: ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers (Tensor device dtype '[batchSize, seqLen, dmodel])
  PosEmbed :: ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers (Tensor device dtype '[batchSize, seqLen, dmodel])
  Blocks :: ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers [V.Vector nLayers (DMap (BlockCache device dtype batchSize seqLen dmodel dhead nhead) Identity)]
  LnFinal :: ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)

deriving instance GEq (ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers)

deriving instance GCompare (ActivationCache device dtype batchSize seqLen dmodel dhead nhead nlayers)

data
  BlockCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    (dhead :: Nat)
    (nhead :: Nat)
    a
  where
  ResidPre :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, dmodel])
  Ln1 :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)
  Attn :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (DMap (AttentionCache device dtype batchSize seqLen dmodel dhead nhead) Identity)
  AttnOut :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, dmodel])
  ResidMid :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, dmodel])
  Ln2 :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity)
  MLP :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (DMap (MLPCache device dtype batchSize seqLen dmodel) Identity)
  MLPOut :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, dmodel])
  ResidPost :: BlockCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, dmodel])

deriving instance GEq (BlockCache device dtype batchSize seqLen dmodel dhead nhead)

deriving instance GCompare (BlockCache device dtype batchSize seqLen dmodel dhead nhead)

data
  AttentionCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    (dhead :: Nat)
    (nhead :: Nat)
    a
  where
  Q :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, nhead, dhead])
  K :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, nhead, dhead])
  V :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, nhead, dhead])
  AttnScores :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, nhead, seqLen, seqLen])
  Pattern :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, nhead, seqLen, seqLen])
  Z :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, nhead, dhead])
  Result :: AttentionCache device dtype batchSize seqLen dmodel dhead nhead (Tensor device dtype '[batchSize, seqLen, nhead, dmodel])

deriving instance GEq (AttentionCache device dtype batchSize seqLen dmodel dhead nhead)

deriving instance GCompare (AttentionCache device dtype batchSize seqLen dmodel dhead nhead)

data
  LayerNormCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
    a
  where
  -- | Scale is the std of the input (residual stream)
  Scale :: LayerNormCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, 1])
  Normalized :: LayerNormCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, dmodel])

deriving instance GEq (LayerNormCache device dtype batchSize seqLen dmodel)

deriving instance GCompare (LayerNormCache device dtype batchSize seqLen dmodel)

data
  MLPCache
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (ffnDim :: Nat)
    a
  where
  -- Right before activation function
  MLPpre :: MLPCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, ffnDim])
  -- Right after activation function
  MLPpost :: MLPCache device dtype batchSize seqLen dmodel (Tensor device dtype '[batchSize, seqLen, ffnDim])

deriving instance GEq (MLPCache device dtype batchSize seqLen dmodel)

deriving instance GCompare (MLPCache device dtype batchSize seqLen dmodel)

-- TODO not sure if we actually need this it's in the example
-- deriveGEq ''CacheTag
-- deriveGCompare ''CacheTag
-- deriveGShow ''CacheTag
-- deriveArgDict ''CacheTag

-- type GPT2BlockCache device batchSize seqLen = DMap (BlockCache device D.Float batchSize seqLen 768 64 12) Identity
-- type GPT2AttentionCache device batchSize seqLen = DMap (AttentionCache device D.Float batchSize seqLen 768 64 12) Identity
-- type GPT2LayerNormCache device batchSize seqLen = DMap (LayerNormCache device D.Float batchSize seqLen 768) Identity
-- type GPT2MLPCache device batchSize seqLen = DMap (MLPCache device D.Float batchSize seqLen 768) Identity
type GPT2ActivationCache device batchSize seqLen = DMap (ActivationCache device D.Float batchSize seqLen 768 64 12 12) Identity

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
    (nhead :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttentionSpec ::
    -- | spec for dropout
    MultiheadAttentionSpec dmodel nhead dtype device
  deriving (Show, Eq)

data
  MultiheadAttention
    (dmodel :: Nat)
    (nhead :: Nat)
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
    MultiheadAttention dmodel nhead dtype device
  deriving (Show, Generic, Parameterized)

multiheadAttention ::
  forall device dtype batchSize seqLen dmodel dhead nhead.
  ( 1 <= nhead,
    dmodel ~ (nhead * dhead),
    All KnownNat '[dmodel, dmodel, dmodel, nhead, seqLen, batchSize, dhead],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | multi-head attention model ADT
  MultiheadAttention dmodel nhead dtype device ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]) ->
  -- | input
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | attention and attention averaged over heads, the `attn_out`.
  ( Tensor device dtype '[batchSize, seqLen, dmodel],
    DMap (AttentionCache device dtype batchSize seqLen dmodel dhead nhead) Identity
  )
multiheadAttention MultiheadAttention {..} attentionMask inp =
  ( attn_out,
    fromList
      [ Q ==> reshape @'[batchSize, seqLen, nhead, dhead] q,
        K ==> reshape @'[batchSize, seqLen, nhead, dhead] k,
        V ==> reshape @'[batchSize, seqLen, nhead, dhead] v,
        AttnScores ==> attn_scores,
        Pattern ==> pattern,
        Z ==> z,
        Result ==> result''
      ]
  )
  where
    -- '[batchSize, nhead, seqLen, dhead]
    scaling = Prelude.sqrt . fromIntegral $ natValI @dhead :: Double
    q = reshape' . forward mhaQInProj $ inp
    k = reshape' . forward mhaKInProj $ inp
    v = reshape' . forward mhaVInProj $ inp
    _maskAttention attentionWeights =
      case attentionMask of
        Nothing -> attentionWeights
        Just am -> attentionWeights `add` unsqueeze @1 am
    attn_scores = _maskAttention $ divScalar scaling $ matmul q (transpose @2 @3 k) -- dot product, scaled (by sqrt, not softmax)
    pattern = softmax @3 attn_scores
    z = transpose @1 @2 $ matmul pattern v
    -- z' is z with the nhead and seqLen swapped so that `matmul z' wo` is broadcasted correctly.
    z' = reshape @'[batchSize, nhead, seqLen, dhead] z
    -- KEY: While `mhaOutProj` is efficiently represented with dims [dmodel, dmodel],
    -- its meaning is clearer when seen as [nhead, dhead, dmodel]. These are equiv because dmodel ~ nhead * dhead
    -- We need to reshape to the meaningful representation in order to calculate result'', which is the meaningful cache.
    wo = reshape @'[nhead, dhead, dmodel] $ toDependent $ weight mhaOutProj
    -- TODO: According to [this](https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/transformer-full-updated.png),
    -- this calculation is very expensive, so much so that TransformerLens has a use_attn_result flag to toggle this.
    result' = matmul z' wo
    result'' = reshape @'[batchSize, seqLen, nhead, dmodel] result'
    attn_out = forward mhaOutProj . reshape @'[batchSize, seqLen, dmodel] $ z
    reshape' ::
      forall seqLen'.
      (KnownNat seqLen') =>
      Tensor device dtype '[batchSize, seqLen', dmodel] ->
      Tensor device dtype '[batchSize, nhead, seqLen', dhead]
    reshape' t' = transpose @1 @2 $ reshape @'[batchSize, seqLen', nhead, dhead] t'

instance
  ( All KnownNat '[dmodel, dmodel, dmodel, nhead],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (MultiheadAttentionSpec dmodel nhead dtype device)
    (MultiheadAttention dmodel nhead dtype device)
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

-- | While most functions directly correspond to its cache map,
-- unfortunately, `transformerMLP` needs to return more than a `MLPCache` because
-- `transformerMLP` is doing a bit too much, it calculates `ln2` as well.
-- I can't modify `MLPCache` because that adheres to the TransformerLens API.
data
  TransformerMLPActivation
    (device :: (D.DeviceType, Nat))
    (dtype :: D.DType)
    (batchSize :: Nat)
    (seqLen :: Nat)
    (dmodel :: Nat)
  = TransformerMLPActivation
  { -- | ln2.normalized
    mlpLn :: DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity,
    mlpCache :: DMap (MLPCache device dtype batchSize seqLen dmodel) Identity,
    mlpOut :: Tensor device dtype '[batchSize, seqLen, dmodel],
    -- | resid_post
    mlpResult :: Tensor device dtype '[batchSize, seqLen, dmodel]
  }

transformerMLP ::
  forall dmodel ffnDim maxSeqLen batchSize dtype device.
  ( BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    All KnownNat '[batchSize, maxSeqLen, dmodel],
    GeluDTypeIsValid device dtype,
    IsSuffixOf '[dmodel] '[batchSize, maxSeqLen, dmodel],
    SumDType dtype ~ dtype,
    AllDimsPositive '[batchSize, maxSeqLen, dmodel],
    KnownDevice device,
    KnownDType dtype,
    MatMulDTypeIsValid device dtype,
    SumDTypeIsValid device dtype,
    MeanDTypeValidation device dtype
  ) =>
  -- | MLP model ADT for transformer
  TransformerMLP dmodel ffnDim dtype device ->
  Tensor device dtype '[batchSize, maxSeqLen, dmodel] -> -- input
  TransformerMLPActivation device dtype batchSize maxSeqLen dmodel -- output
transformerMLP TransformerMLP {..} x = TransformerMLPActivation {mlpLn = mlpLn, mlpCache = mlpCache, mlpOut = linear1Out, mlpResult = residPost}
  where
    mlpLn = layerNormForwardCached ln x
    linear0Out = forward linear0 (runIdentity $ mlpLn ! Normalized)
    actFnOut = linear0Out `geluApproximate` "tanh"
    linear1Out = forward linear1 actFnOut
    residPost = x `add` linear1Out
    mlpCache = fromList [MLPpre ==> linear0Out, MLPpost ==> actFnOut]

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
    (nhead :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayerSpec ::
    forall dmodel nhead ffnDim dtype device.
    { mhaSpec :: MultiheadAttentionSpec dmodel nhead dtype device,
      epsSpec' :: Double,
      mlpSpec :: TransformerMLPSpec dmodel ffnDim dtype device
    } ->
    TransformerLayerSpec dmodel nhead ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerLayer
    (dmodel :: Nat)
    (nhead :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayer ::
    forall dmodel nhead ffnDim dtype device.
    { -- | multi-head attention
      transformerLayer_mha :: MultiheadAttention dmodel nhead dtype device,
      -- | layer norm
      transformerLayer_ln :: LayerNorm '[dmodel] dtype device,
      -- | MLP
      transformerLayer_mlp :: TransformerMLP dmodel ffnDim dtype device
    } ->
    TransformerLayer dmodel nhead ffnDim dtype device
  deriving (Show, Generic, Parameterized)

std ::
  forall dim keepOrDropDim shape' shape dtype device.
  ( KnownNat dim,
    KnownKeepOrDropDim keepOrDropDim,
    shape' ~ ConditionalDropDimension shape dim keepOrDropDim,
    MeanDTypeValidation device dtype,
    AllDimsPositive shape
  ) =>
  Tensor device dtype shape ->
  Tensor device dtype shape'
std input =
  unsafePerformIO $
    ATen.cast3
      ATen.Managed.std_tlb
      input
      (natValI @dim)
      (keepOrDropDimVal @keepOrDropDim)

layerNormForwardCached ::
  forall (dmodel :: Nat) (seqLen :: Nat) (batchSize :: Nat) dtype device.
  ( All KnownNat '[batchSize, seqLen, dmodel],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    MeanDTypeValidation device dtype,
    AllDimsPositive '[batchSize, seqLen, dmodel]
  ) =>
  LayerNorm '[dmodel] dtype device ->
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  DMap (LayerNormCache device dtype batchSize seqLen dmodel) Identity
layerNormForwardCached ln inp =
  fromList
    [ Scale ==> scale,
      Normalized ==> result
    ]
  where
    result = forward ln inp
    -- The standard deviation over each token
    scale = std @2 @KeepDim @'[batchSize, seqLen, 1] @'[batchSize, seqLen, dmodel] @dtype @device inp

transformerLayer ::
  forall (nhead :: Nat) (ffnDim :: Nat) (dmodel :: Nat) (dhead :: Nat) (seqLen :: Nat) (batchSize :: Nat) dtype device.
  ( 1 <= nhead,
    dmodel ~ (dhead * nhead),
    All KnownNat '[dmodel, dmodel, dmodel, nhead, seqLen, batchSize, dhead],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    KnownDType dtype,
    dtype ~ SumDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    GeluDTypeIsValid device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device,
    MeanDTypeValidation device dtype,
    AllDimsPositive '[batchSize, seqLen, dmodel]
  ) =>
  -- | transformer layer model ADT
  TransformerLayer dmodel nhead ffnDim dtype device ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[batchSize, seqLen, seqLen]) ->
  -- | input
  Tensor device dtype '[batchSize, seqLen, dmodel] ->
  -- | transformer layer output representation
  (Tensor device dtype '[batchSize, seqLen, dmodel], DMap (BlockCache device dtype batchSize seqLen dmodel dhead nhead) Identity)
transformerLayer TransformerLayer {..} attentionMask inp =
  ( residPost,
    fromList
      [ ResidPre ==> inp,
        Ln1 ==> lnCache,
        Attn ==> attnCache,
        AttnOut ==> attnOut,
        ResidMid ==> residMid,
        Ln2 ==> ln2,
        MLP ==> mlpCache,
        MLPOut ==> mlpOut,
        ResidPost ==> residPost
      ]
  )
  where
    lnCache = layerNormForwardCached transformerLayer_ln inp
    (attnOut, attnCache) = multiheadAttention transformerLayer_mha attentionMask (runIdentity $ lnCache ! Normalized)
    residMid = inp `add` attnOut
    TransformerMLPActivation ln2 mlpCache mlpOut residPost = transformerMLP transformerLayer_mlp residMid

instance
  ( All KnownNat '[dmodel, dmodel, dmodel, nhead, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerLayerSpec dmodel nhead ffnDim dtype device)
    (TransformerLayer dmodel nhead ffnDim dtype device)
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
    (nhead :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (dmodel :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2Spec ::
    forall numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device.
    { -- | spec for each and every transformer layer
      lmLayerSpec :: TransformerLayerSpec dmodel nhead ffnDim dtype device,
      epsSpec'' :: Double
    } ->
    GPT2Spec numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device
  deriving (Show, Eq)

data
  GPT2
    (numAttnLayers :: Nat)
    (nhead :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (dmodel :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2 ::
    forall numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device.
    { -- | token embedding
      tEmbedding :: Embedding ('Just paddingIdx) vocabSize dmodel 'Learned dtype device,
      -- | positional embedding
      tPosEmbedding :: Embedding 'Nothing maxSeqLen dmodel 'Constant dtype device,
      -- | transformer layers
      tLayers :: HList (HReplicateR numAttnLayers (TransformerLayer dmodel nhead ffnDim dtype device)),
      -- | final layer norm
      tFinalLN :: LayerNorm '[dmodel] dtype device,
      -- | final output projection
      tProj :: Linear dmodel vocabSize dtype device
    } ->
    GPT2 numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device
  deriving (Generic)

deriving instance
  ( Show
      ( HList
          ( HReplicateR
              numAttnLayers
              ( TransformerLayer
                  dmodel
                  nhead
                  ffnDim
                  dtype
                  device
              )
          )
      )
  ) =>
  Show (GPT2 numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)

instance
  ( layers
      ~ HReplicateR
          numAttnLayers
          ( TransformerLayer
              dmodel
              nhead
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
  Parameterized (GPT2 numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)

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
  ( 1 <= nhead,
    dmodel ~ (dhead * nhead),
    All KnownNat '[dmodel, nhead, seqLen, batchSize, dhead],
    IsSuffixOf '[dmodel] '[batchSize, seqLen, dmodel],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    GeluDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device,
    MeanDTypeValidation device dtype,
    AllDimsPositive '[batchSize, seqLen, dmodel]
  ) =>
  ApplyAB
    (FoldLayers batchSize seqLen dtype device)
    ( TransformerLayer dmodel nhead ffnDim dtype device,
      Tensor device dtype '[batchSize, seqLen, dmodel]
    )
    ( Tensor device dtype '[batchSize, seqLen, dmodel],
      DMap
        (BlockCache device dtype batchSize seqLen dmodel dhead nhead)
        Identity
    )
  where
  applyAB FoldLayers {..} (layer, x) = do
    let (res, cache) = transformerLayer layer flAttentionMask x in (res, cache)

transformerLM ::
  forall
    numAttnLayers
    nhead
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
      (HReplicateR numAttnLayers (TransformerLayer dmodel nhead ffnDim dtype device))
      (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  GPT2 numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device ->
  Tensor device 'D.Int64 '[batchSize, seqLen] ->
  (Tensor device dtype '[batchSize, seqLen, vocabSize], GPT2ActivationCache device batchSize seqLen)
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
      (HReplicateR numAttnLayers (TransformerLayer dmodel nhead ffnDim dtype device))
      (HReplicateR numAttnLayers (Tensor device dtype '[batchSize, seqLen, dmodel])),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  HasForward (GPT2 numAttnLayers nhead ffnDim paddingIdx seqLen vocabSize dmodel dtype device) (Tensor device 'D.Int64 '[batchSize, seqLen]) (Tensor device dtype '[batchSize, seqLen, vocabSize])
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
    HReplicate numAttnLayers (TransformerLayerSpec dmodel nhead ffnDim dtype device),
    A.Randomizable
      (HList (HReplicateR numAttnLayers (TransformerLayerSpec dmodel nhead ffnDim dtype device)))
      (HList (HReplicateR numAttnLayers (TransformerLayer dmodel nhead ffnDim dtype device))),
    KnownDType dtype,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  A.Randomizable
    (GPT2Spec numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)
    (GPT2 numAttnLayers nhead ffnDim paddingIdx maxSeqLen vocabSize dmodel dtype device)
  where
  sample GPT2Spec {..} =
    GPT2
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample (LayerNormSpec epsSpec'')
      <*> A.sample LinearSpec
